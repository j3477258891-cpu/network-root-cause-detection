import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { Workbook } from '@oai/artifact-tool';

// Only this builder authors submission CSVs. Python supplies the frozen spec.
const spec = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const source = await Workbook.fromCSV(await fs.readFile(spec.base_file, 'utf8'), {sheetName:'Submission'});
const sheet = source.worksheets.getItemAt(0);
const rows = sheet.getUsedRange().values;
const byOrder = new Map(rows.slice(1).map((r, i) => [r[0], i+1]));
for (const c of spec.additions) {
  const index = byOrder.get(c.order_id);
  if (index === undefined) throw new Error('Unknown order');
  const obj = JSON.parse(rows[index][1]);
  if (obj.rootcause.some(x=>x['@rid']===c.add_rid)) throw new Error('Duplicate addition');
  obj.rootcause.push(c.node);
  if(obj.rootcause.length>8) throw new Error('Too many roots');
  rows[index][1] = JSON.stringify(obj);
}
sheet.getRange(`A1:B${rows.length}`).values = rows;
const authored = sheet.getUsedRange().values;
const escape = x => /[",\r\n]/.test(String(x)) ? `"${String(x).replaceAll('"','""')}"` : String(x);
const bytes = Buffer.from(authored.map(r=>r.map(escape).join(',')).join('\r\n')+'\r\n','utf8');
try { await fs.access(spec.output_file); throw new Error('Refusing to overwrite an existing CSV'); }
catch(e) { if(e.code!=='ENOENT') throw e; }
await fs.writeFile(spec.output_file, bytes, {flag:'wx'});
const imported = await Workbook.fromCSV(bytes.toString('utf8'), {sheetName:'Submission'});
const reloaded = imported.worksheets.getItemAt(0).getUsedRange().values;
if(JSON.stringify(reloaded)!==JSON.stringify(authored)) throw new Error('CSV roundtrip mismatch');
const inspection=await imported.inspect({kind:'table',sheetId:'Submission',range:'A1:B4',tableMaxRows:4,tableMaxCols:2,maxChars:1400});
if(!inspection.ndjson.includes('order_id')) throw new Error('Artifact inspection failed');
const errorScan=await imported.inspect({kind:'match',sheetId:'Submission',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A',options:{useRegex:true,maxResults:10},maxChars:800});
if(/"matchCount"\s*:\s*[1-9]/.test(errorScan.ndjson)) throw new Error('Formula error tokens');
// A readable field-level preview of changed rows. The raw JSON CSV is checked
// in full above; this in-memory view neither modifies nor replaces that CSV.
const view=Workbook.create();
const ps=view.worksheets.add('Changes');
const previewRows=[['order_id','rootcause_count','added_rid'],...spec.additions.slice(0,8).map(c=>{
  const row=reloaded[byOrder.get(c.order_id)];
  return [c.order_id,JSON.parse(row[1]).rootcause.length,c.add_rid];
})];
ps.getRange(`A1:C${previewRows.length}`).values=previewRows;
ps.getRange('A1:A9').format.columnWidthPx=310;
ps.getRange('B1:B9').format.columnWidthPx=145;
ps.getRange('C1:C9').format.columnWidthPx=380;
ps.getRange('A1:C9').format.rowHeightPx=32;
ps.getRange('A1:C1').format.font.bold=true;
const preview=await view.render({sheetName:'Changes',range:`A1:C${previewRows.length}`,scale:1,format:'png'});
await fs.mkdir(path.dirname(spec.preview_file),{recursive:true});
await fs.writeFile(spec.preview_file,new Uint8Array(await preview.arrayBuffer()));
console.log(JSON.stringify({file:spec.output_file,sha256:crypto.createHash('sha256').update(bytes).digest('hex'),
  artifact_tool_roundtrip:true,artifact_tool_inspected:true,preview_file:spec.preview_file}));
