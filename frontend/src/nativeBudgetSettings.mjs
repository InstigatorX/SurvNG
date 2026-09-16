export const BUDGET_FIELDS = [
  {key:'idle_fps', label:'Idle detection FPS', initial:1, min:.5, max:10, step:.5},
  {key:'active_fps', label:'Active detection FPS', initial:5, min:.5, max:10, step:.5},
  {key:'cooldown_seconds', label:'Cooldown (seconds)', initial:5, min:0, max:300, step:.5},
  {key:'approach_padding', label:'Approach padding (fraction of frame)', initial:.1, min:0, max:.5, step:.01},
  {key:'block_size', label:'Block size (pixels)', initial:64, min:16, max:512, step:1, motion:true},
  {key:'motion_threshold', label:'Motion threshold', initial:.05, min:0, max:1, step:.01, motion:true},
  {key:'min_persistence', label:'Minimum persistence (frames)', initial:2, min:1, max:30, step:1, motion:true},
  {key:'max_miss', label:'Maximum missed frames', initial:1, min:0, max:30, step:1, motion:true},
  {key:'iou_threshold', label:'Motion matching IoU threshold', initial:.3, min:0, max:1, step:.01, motion:true},
  {key:'smooth_alpha', label:'Motion smoothing alpha', initial:.5, min:0, max:1, step:.01, motion:true},
  {key:'confirm_frames', label:'Motion confirmation frames', initial:1, min:1, max:10, step:1, motion:true},
  {key:'pixel_diff_threshold', label:'Pixel difference threshold', initial:15, min:1, max:255, step:1, motion:true},
  {key:'min_rel_area', label:'Minimum relative motion area', initial:.0005, min:0, max:.25, step:.0001, motion:true},
];
export function effectiveBudget(values={}, defaults={}) {
  return Object.fromEntries([['enabled', false],['motion_enabled',true], ...BUDGET_FIELDS.map(f=>[f.key,f.initial])]
    .map(([key,initial])=>[key, values[key] ?? defaults[key] ?? initial]));
}
