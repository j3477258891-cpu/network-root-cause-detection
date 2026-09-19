import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import path from 'node:path';
import {Workbook} from '@oai/artifact-tool';

// Only this Artifact Tool builder writes actual submission CSVs.
const spec=JSON.parse(await fs.readFile(process.argv[2],'utf8'));
const base=await fs.readFile(spec.base_file);
if(crypto.createHash('sha256').update(base).digest('hex')!==spec.base_sha256) throw Error('Frozen baseline changed');
const wb=await Workbook.fromCSV(base.toString('utf8'),{sheetName:'Submission'});
const sheet=wb.worksheets.getItemAt(0);
const rows=sheet.getUsedRange().values;
if(rows.length!==547 || rows[0].join('|')!=='order_id|output') throw Error('Baseline structure changed');
const byOrder=new Map(rows.slice(1).map((r,i)=>[r[0],i+1]));
const seen=new Set();
for(const a of spec.actions){
  if(seen.has(a.order_id))throw Error('Multiple actions in one order');
  seen.add(a.order_id);
  const index=byOrder.get(a.order_id);
  if(index===undefined)throw Error('Unknown order');
  const data=JSON.parse(rows[index][1]);
  if(a.remove_rid){
    const hits=data.rootcause.filter(n=>n['@rid']===a.remove_rid);
    if(hits.length!==1)throw Error('Missing/duplicate removal');
    data.rootcause=data.rootcause.filter(n=>n['@rid']!==a.remove_rid);
  }
  if(a.add_rid){
    if(data.rootcause.some(n=>n['@rid']===a.add_rid)||a.node['@rid']!==a.add_rid)throw Error('Invalid addition');
    data.rootcause.push(a.node);
  }
  if(data.rootcause.length<1||data.rootcause.length>8)throw Error('Root count outside 1..8');
  rows[index][1]=JSON.stringify(data);
}
sheet.getRange('A1:B547').values=rows;
const values=sheet.getUsedRange().values;
const escape=x=>/[",\r\n]/.test(String(x))?`"${String(x).replaceAll('"','""')}"`:String(x);
const bytes=Buffer.from(values.map(r=>r.map(escape).join(',')).join('\r\n')+'\r\n','utf8');
const reload=await Workbook.fromCSV(bytes.toString('utf8'),{sheetName:'Submission'});
if(JSON.stringify(reload.worksheets.getItemAt(0).getUsedRange().values)!==JSON.stringify(values))throw Error('CSV roundtrip mismatch');
const inspection=await reload.inspect({kind:'table',sheetId:'Submission',range:'A1:B4',tableMaxRows:4,tableMaxCols:2,maxChars:1200});
if(!inspection.ndjson.includes('order_id'))throw Error('Artifact inspection failed');
await reload.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options:{useRegex:true,maxResults:30},summary:'Submission literal/formula error scan',maxChars:1200});
const errorToken=/^#(?:REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!|SPILL!|CALC!)$/;
if(values.some(r=>r.some(v=>errorToken.test(String(v)))))throw Error('Error cell in exported values');
const view=Workbook.create();const ps=view.worksheets.add('Changes');
const display=[['order_id','action','root_count'],...spec.actions.slice(0,8).map(a=>[a.order_id,a.kind,JSON.parse(rows[byOrder.get(a.order_id)][1]).rootcause.length])];
ps.getRange(`A1:C${display.length}`).values=display;
ps.getRange(`A1:A${display.length}`).format.columnWidthPx=350;
ps.getRange(`B1:C${display.length}`).format.columnWidthPx=125;
ps.getRange(`A1:C${display.length}`).format.rowHeightPx=30;
ps.getRange('A1:C1').format.font.bold=true;
const preview=await view.render({sheetName:'Changes',range:`A1:C${display.length}`,scale:1,format:'png'});
await fs.mkdir(path.dirname(spec.preview_file),{recursive:true});
await fs.writeFile(spec.preview_file,new Uint8Array(await preview.arrayBuffer()));
// Do not overwrite a scored or earlier generated file; publish only after roundtrip/render validation.
await fs.writeFile(spec.output_file,bytes,{flag:'wx'});
console.log(JSON.stringify({file:spec.output_file,sha256:crypto.createHash('sha256').update(bytes).digest('hex'),
  roundtrip:true,inspected:true,error_scan:true,preview_file:spec.preview_file}));
