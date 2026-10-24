import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {waitForRainfallWriter} from './rainfall_refresh_poll.mjs';
const origin='https://fixture.invalid',sleep=async()=>{};
test('busy writer is read until ownership is released, without any POST',async()=>{
 let calls=0;const result=await waitForRainfallWriter({origin,sleep,fetcher:async(url,o)=>{assert.equal(o.method,undefined);return Response.json({updater:{state:'running',cursor:++calls,writer:{active:calls<3}}});}});
 assert.equal(result.ready,true);assert.equal(result.cursor,3);assert.equal(calls,3);
});
test('scientific failures stop retries even when the writer is inactive',async()=>{
 await assert.rejects(()=>waitForRainfallWriter({origin,sleep,fetcher:async()=>Response.json({updater:{state:'failed',writer:{active:false},failure_detail:{reason:'Changed source SHA'}}})}),/Changed source SHA/);
});
test('timeout, missing status and unknown owner never authorize a second writer',async()=>{
 for(const updater of [null,{state:'running',writer:{active:true}},{state:'running',writer:{}}]){
  const r=await waitForRainfallWriter({origin,sleep,maxPolls:2,fetcher:async()=>Response.json({updater})});assert.equal(r.ready,false);assert.equal(r.recovery_pending,true);
 }
 const r=await waitForRainfallWriter({origin,sleep,now:()=>20,deadline:10,fetcher:()=>{throw Error('Must not fetch after deadline');}});assert.equal(r.ready,false);
});
test('transient read errors are bounded and an expired/closed writer can be resumed',async()=>{
 let n=0;const r=await waitForRainfallWriter({origin,sleep,fetcher:async()=>{if(++n===1)throw Error('ECONNRESET');if(n===2)return new Response(null,{status:503});return Response.json({updater:{state:'interrupted',cursor:3,writer:{active:false,expired:true}}});}});assert.equal(r.ready,true);assert.equal(r.cursor,3);
});
test('unknown scientific state is never retried and freshness checks remain enforced',async()=>{
 await assert.rejects(()=>waitForRainfallWriter({origin,sleep,fetcher:async()=>Response.json({updater:{state:'unknown',writer:{active:false}}})}),/Unknown/);
 const source=fs.readFileSync(new URL('../.github/workflows/rainfall-v1314b-live.yml',import.meta.url),'utf8');
 assert(source.includes('now-anchor<=12*3600000'));assert(source.includes('record.sha256'));assert(!source.includes('continue-on-error'));assert(source.includes('waitForRainfallWriter'));
});
