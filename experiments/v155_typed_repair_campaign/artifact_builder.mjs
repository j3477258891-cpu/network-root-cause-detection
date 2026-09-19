// Reuse the historical, hash-protected ArtifactTool CSV writer. The supplied
// authoring spec points exclusively at the new V155 output and p03 baseline.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const spec=JSON.parse(await fs.readFile(process.argv[2],'utf8'));
if(path.dirname(path.resolve(spec.output_file))!==path.resolve(here))throw Error('V155 output must remain inside campaign');
if(!path.basename(spec.output_file).startsWith('v155_'))throw Error('Wrong campaign prefix');
if(spec.base_sha256!=='102b99385a8094a1d47b2748dfaa79fbd5a2b2b69936dffa9e5e7d9636fba196')throw Error('Wrong p03 baseline');
if(!spec.actions.length)throw Error('Do not emit a no-op champion copy');
await import('../v153_complementary_repair_campaign/artifact_builder.mjs');
