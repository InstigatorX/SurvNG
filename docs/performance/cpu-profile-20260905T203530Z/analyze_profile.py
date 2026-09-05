#!/usr/bin/env python3
"""Analyze saved stacks and independently measured thread CPU; no attachment."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/root/SurvNG/survng/app/motion_pipeline/adaptive_stages.py')


def main():
    raw = json.loads((ROOT / 'pyspy-native.speedscope.json').read_text())
    cpu = json.loads((ROOT / 'cpu-summary.json').read_text())
    frames = raw['shared']['frames']
    own, inclusive, sites = collections.Counter(), collections.Counter(), collections.Counter()
    paths, stage_counts, stage_self, stage_selection = (collections.Counter() for _ in range(4))
    classes = [(node.lineno, node.end_lineno, node.name) for node in ast.parse(SOURCE.read_text()).body
               if isinstance(node, ast.ClassDef)]
    threads, categories = [], collections.Counter()
    total = 0
    total_weight = 0.0
    copy_count = cv_count = selection_count = 0
    wait_symbols = {'pthread_cond_timedwait', 'pthread_cond_wait', 'pthread_mutex_lock',
                    'pthread_rwlock_rdlock', 'pthread_rwlock_wrlock', 'sem_wait', 'sem_timedwait'}
    io_symbols = {'select', 'poll', 'epoll_wait', 'epoll_pwait', 'read', 'pread64', 'write',
                  'recv', 'recvfrom', 'recvmsg', 'accept', 'accept4'}
    for profile in raw['profiles']:
        tid = int(re.search(r'Thread (\d+)', profile['name']).group(1))
        threads.append({'tid': tid, 'name': profile['name'], 'samples': len(profile['samples']),
                        'weight_sum_seconds': sum(profile.get('weights', []))})
        for sample, weight in zip(profile['samples'], profile['weights']):
            if not sample:
                continue
            total += 1
            total_weight += weight
            stack = [frames[index] for index in sample]
            leaf = stack[-1]
            key = (leaf['name'], leaf.get('file', ''))
            own[key] += 1
            sites[(leaf['name'], leaf.get('file', ''), leaf.get('line', 0))] += 1
            keys = {(frame['name'], frame.get('file', '')) for frame in stack}
            inclusive.update(keys)
            symbols = {frame['name'] for frame in stack}
            selection = leaf['name'].startswith('introselect_')
            if symbols & wait_symbols:
                category = 'synchronization_or_wait_stack'
            elif symbols & io_symbols:
                category = 'io_or_polling_stack'
            elif selection:
                category = 'numpy_selection_computation_leaf'
            else:
                category = 'other_or_unresolved'
            categories[category] += 1
            selection_count += int(selection)
            copy_count += int(any(name in symbols for name in ('PyArray_CopyAsFlat', 'PyArray_CopyInto',
                                                               'PyArray_NewCopy', 'array_copy', 'memcpy', 'memmove')))
            cv_count += int(any('/cv2/' in frame.get('file', '') for frame in stack))
            app = [frame for frame in stack if frame.get('file', '').startswith('/root/SurvNG/survng/')]
            if app:
                frame = app[-1]
                paths[(frame['file'], frame['name'], frame.get('line', 0))] += 1
            seen_classes = set()
            for frame in stack:
                if frame.get('file') == str(SOURCE):
                    class_name = next((name for lower, upper, name in classes
                                       if lower <= frame.get('line', 0) <= upper), 'unknown')
                    seen_classes.add(class_name)
                    if selection:
                        stage_selection[(class_name, frame.get('line', 0))] += 1
            for class_name in seen_classes:
                stage_counts[class_name] += 1
                stage_self[(class_name, category)] += 1

    def rows(counter):
        return [{'name': key[0], 'file': key[1], 'samples': count,
                 'percent_of_all_sampled_stacks': count * 100 / total}
                for key, count in counter.most_common()]

    tids = {row['tid'] for row in threads}
    tid_names = {row['tid']: row['name'] for row in threads}
    represented = [row for row in cpu['threads'] if row['tid'] in tids]
    unrepresented = [row for row in cpu['threads'] if row['tid'] not in tids]
    main_cpu = cpu['category_cpu_seconds']['main']
    groups = collections.Counter()
    for row in cpu['threads']:
        name = tid_names.get(row['tid'], '')
        group = ('motion-analysis' if 'motion-analysis-' in name else
                 'recording-indexer' if 'recording-indexer' in name else
                 'recording-index-maintenance' if 'recording-index-maintenance' in name else
                 'other_represented' if name else 'not_represented_in_profile')
        groups[group] += row['cpu_seconds']
    summary = {
        'profile_sha256': hashlib.sha256((ROOT / 'pyspy-native.speedscope.json').read_bytes()).hexdigest(),
        'profile_thread_count': len(threads), 'sampled_stack_count': total,
        'sample_weight_sum_seconds_not_cpu': total_weight, 'tool_reported_errors': 11,
        'self_functions': rows(own), 'inclusive_functions': rows(inclusive),
        'self_sites': [{'name': key[0], 'file': key[1], 'line': key[2], 'samples': value}
                       for key, value in sites.most_common()],
        'innermost_application_paths': [{'file': key[0], 'name': key[1], 'line': key[2], 'samples': value}
                                        for key, value in paths.most_common()],
        'stage_inclusive_stacks': dict(stage_counts),
        'stage_stack_categories': [{'stage': key[0], 'category': key[1], 'samples': value}
                                   for key, value in stage_self.most_common()],
        'selection_computation_callers': [{'stage': key[0], 'line': key[1], 'self_samples': value}
                                          for key, value in stage_selection.most_common()],
        'stack_categories': dict(categories),
        'copy_api_inclusive_samples': copy_count, 'cv2_library_inclusive_samples': cv_count,
        'selection_computation_self_samples': selection_count,
        'threads': sorted(threads, key=lambda row: row['samples'], reverse=True),
        'cpu_coverage': {
            'main_cpu_seconds': main_cpu,
            'represented_thread_cpu_seconds': sum(row['cpu_seconds'] for row in represented),
            'represented_thread_percent_main_cpu': sum(row['cpu_seconds'] for row in represented) / main_cpu * 100,
            'unrepresented_thread_cpu_seconds': sum(row['cpu_seconds'] for row in unrepresented),
            'unrepresented_thread_percent_main_cpu': sum(row['cpu_seconds'] for row in unrepresented) / main_cpu * 100,
            'main_minus_thread_counter_seconds': cpu['main_process_minus_observed_threads_cpu_seconds'],
            'named_group_cpu_seconds': dict(groups), 'unrepresented_threads': unrepresented,
            'note': 'Thread identity coverage only; py-spy stack fractions are not exact per-function CPU shares. Polling, waits, native-only workers, setup/write boundaries and scheduler quantization limit attribution.'},
        'source_read_after_profile': {'path': str(SOURCE), 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        'perf_availability': {
            'path': shutil.which('perf'),
            'checked_executable_paths': {'/usr/bin/perf': Path('/usr/bin/perf').is_file(),
                                         '/usr/local/bin/perf': Path('/usr/local/bin/perf').is_file()},
            'linux_tools_directories': [str(path) for path in Path('/usr/lib').glob('linux-tools*')],
            'second_window': 'Not run: perf executable unavailable; installation/kernel changes prohibited.'},
    }
    (ROOT / 'profile-analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in ('sampled_stack_count', 'stack_categories',
          'stage_inclusive_stacks', 'copy_api_inclusive_samples', 'cv2_library_inclusive_samples',
          'selection_computation_callers', 'perf_availability')}, indent=2))


if __name__ == '__main__':
    main()
