import React from 'react';
import { BUDGET_FIELDS, effectiveBudget } from '../nativeBudgetSettings.mjs';

export default function NativeBudgetSettings({ values={}, defaults={}, overrides=false, onChange }) {
  const effective = effectiveBudget(values, defaults);
  const inherited = effectiveBudget(defaults);
  const toggle = (key, label) => overrides
    ? <label>{label}<select aria-label={label} value={values[key] == null ? '' : String(values[key])} onChange={e=>onChange(key,e.target.value === '' ? null : e.target.value === 'true')}>
        <option value="">Use global ({inherited[key] ? 'On' : 'Off'})</option><option value="true">On</option><option value="false">Off</option>
      </select></label>
    : <label className="check-field"><input type="checkbox" checked={effective[key]} onChange={e=>onChange(key,e.target.checked)} /> {label}</label>;
  const fields = motion => <div className="form-grid">{BUDGET_FIELDS.filter(f=>Boolean(f.motion)===motion).map(f=><label key={f.key}>{f.label}<input aria-label={f.label} type="number" min={f.min} max={f.max} step={f.step}
    value={overrides ? values[f.key] ?? '' : values[f.key] ?? f.initial} placeholder={overrides ? `Global: ${inherited[f.key]}` : undefined}
    onChange={e=>onChange(f.key,e.target.value === '' ? (overrides ? null : '') : Number(e.target.value))} /></label>)}</div>;
  return <section className="sub-panel">
    <h3>Adaptive inference budget{overrides ? ' overrides' : ''}</h3>
    {toggle('enabled','Adaptive inference enabled')}
    <p>Quiet cameras keep full-frame detection at the idle rate. Relevant motion or fresh objects near incident zones switch to the active rate. These are fresh detections per second; fixed-mode inference interval does not apply. Ignore zones suppress incidents without removing detection coverage.</p>
    {overrides ? <p>Leave values blank to inherit the global settings.</p> : null}
    {fields(false)}
    {effective.idle_fps > effective.active_fps ? <p role="alert">Idle FPS must not exceed active FPS.</p> : null}
    <small>Confirmation gets a minimum active hold even with zero cooldown. Larger approach padding wakes earlier. Brief appearances can still be missed between idle checks.</small>
    <details><summary>Native motion wake-up (gvamotiondetect)</summary>
      {toggle('motion_enabled','Motion wake-up enabled')}
      <p>Motion wakes inference; it cannot create or extend incidents. Use “Exclude from motion wake-up” in zone settings to suppress movement in selected areas. Raise motion/pixel thresholds, confirmation frames, or minimum area to reduce foliage and lighting triggers. Higher persistence delays wake-up.</p>
      {fields(true)}
    </details>
  </section>;
}
