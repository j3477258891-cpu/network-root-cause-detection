import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { Workbook } from '@oai/artifact-tool';

if (process.argv[2] === '--help-csv') {
  const w=Workbook.create();
  console.log(w.help('CSV export', {search:'csv|CSV',include:'index,examples,notes',maxChars:4500}).ndjson);
  process.exit(0);
}
const spec=JSON.parse(await fs.readFile(process.argv[2],'utf8'));
const original=await fs.readFile(spec.baseline.file);
assert.equal(crypto.createHash('sha256').update(original).digest('hex'),spec.baseline.sha256);
assert(!original.subarray(0,3).equals(Buffer.from([239,187,191])));
const wb=await Workbook.fromCSV(original.toString('utf8'),{sheetName:'Submission'});
const sh=wb.worksheets.getItem('Submission');
const data=sh.getRange('A1:B547').values;
assert.deepEqual(data[0],['order_id','output']);
assert.equal(new Set(data.slice(1).map(r=>r[0])).size,546);
const cat=JSON.parse(await fs.readFile(spec.catalog,'utf8')).candidates;
const seen=new Set();
for(const id of spec.query_ids){
  const a=cat.find(x=>x.candidate_id===id);assert(a);
  assert(!seen.has(a.order_id));seen.add(a.order_id);
  const index=data.findIndex(r=>r[0]===a.order_id);assert(index>0);
  const obj=JSON.parse(data[index][1]);let roots=obj.rootcause;
  if(a.remove_rid){assert.equal(roots.filter(n=>n['@rid']===a.remove_rid).length,1);roots=roots.filter(n=>n['@rid']!==a.remove_rid);}
  if(a.add_rid){assert(!roots.some(n=>n['@rid']===a.add_rid));roots.push(a.node);}
  assert(roots.length>=1&&roots.length<=8);obj.rootcause=roots;
  sh.getCell(index,1).values=[[JSON.stringify(obj)]];
}
wb.recalculate();
const output=sh.getRange('A1:B547').values;
let count=0;
for(const row of output.slice(1)){
  const nodes=JSON.parse(row[1]).rootcause;count+=nodes.length;
  assert(nodes.length>=1&&nodes.length<=8);assert.equal(new Set(nodes.map(n=>n['@rid'])).size,nodes.length);
}
assert.equal(count,spec.expected_p);
console.log((await wb.inspect({kind:'region',sheetId:'Submission',range:'A1:B2',maxChars:400,tableMaxCellChars:60})).ndjson);
// RFC 4180 serialization of the authored worksheet values preserves JSON fields
// and the competition's exact two-column, no-BOM, CRLF format.
const quote=x=>/[",\r\n]/.test(String(x))?'"'+String(x).replaceAll('"','""')+'"':String(x);
const csv=output.map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n';
const roundtrip=await Workbook.fromCSV(csv,{sheetName:'Roundtrip'});
assert.deepEqual(roundtrip.worksheets.getItem('Roundtrip').getRange('A1:B547').values,output);
await fs.writeFile(spec.output,csv,{encoding:'utf8',flag:'wx'});
console.log(JSON.stringify({file:spec.output,predictions:count,orders:546,encoding:'utf-8-no-bom'}));
