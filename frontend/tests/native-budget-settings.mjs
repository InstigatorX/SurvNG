import assert from 'node:assert/strict';
import {effectiveBudget,BUDGET_FIELDS} from '../src/nativeBudgetSettings.mjs';
assert.equal(effectiveBudget().enabled,false);
assert.equal(effectiveBudget({enabled:false},{enabled:true}).enabled,false);
assert.equal(effectiveBudget({enabled:null},{enabled:true}).enabled,true);
assert.equal(effectiveBudget({idle_fps:null},{idle_fps:2}).idle_fps,2);
assert.equal(effectiveBudget({motion_threshold:.3},{motion_threshold:.2}).motion_threshold,.3);
assert.equal(BUDGET_FIELDS.filter(f=>f.motion).length,9);
console.log('Budget inheritance controls passed');
