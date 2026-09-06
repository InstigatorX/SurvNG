export function modelEvaluationCoverage(result) {
  if (!result) return null;
  const total = Number(result.sample_count || 0);
  const baselineErrors = Number(result.baseline?.errors || 0);
  const candidateErrors = Number(result.candidate?.errors || 0);
  const compared = Number(result.compared_sample_count ?? (baselineErrors || candidateErrors ? 0 : total));
  const failed = Number(result.failed_sample_count ?? Math.max(0, total - compared));
  return {
    total,
    compared,
    failed,
    baselineErrors,
    candidateErrors,
    complete: compared > 0 && failed === 0 && baselineErrors === 0 && candidateErrors === 0,
  };
}
