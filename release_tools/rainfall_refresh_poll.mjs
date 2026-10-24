// Read-only coordination: an active hosted writer is not a failed forecast.
// Never issue a POST here, and never authorize retry after a scientific failure.
export async function waitForRainfallWriter({origin,fetcher=fetch,sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms)),now=Date.now,deadline=now()+7*60*1000,maxPolls=210}={}){
 for(let poll=0;poll<maxPolls&&now()<deadline;poll++){
  await sleep(2000);
  let data;
  try{const r=await fetcher(origin+'/api/rainfall/live?model=v1315',{signal:AbortSignal.timeout(30000),cache:'no-store'});if(!r.ok)continue;data=await r.json();}catch{continue;}
  const s=data?.updater;
  if(s?.state==='failed'||s?.job_state==='failed')throw Error(s.failure_detail?.reason||s.error||'Stored scientific stage failed; preserve for inspection');
  if(!s||s.writer?.active!==false)continue;
  if(['complete','running','interrupted','waiting_for_source'].includes(s.state))return {ready:true,id:s.id,cursor:s.cursor,state:s.state};
  throw Error('Unknown stored writer outcome; no blind retry');
 }
 return {ready:false,busy:true,recovery_pending:true};
}
