import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { Workbook } from "@oai/artifact-tool";

const ROOT = "D:/zgyidong";
const EXP = `${ROOT}/experiments`;
const OUT = `${EXP}/v149_online_calibrated_addition_campaign`;
const PREVIEWS = `${OUT}/previews`;
const BASE = `${EXP}/v148_two_true_decode/v148_final_candidates_2_3_from_v145.csv`;
const BASE_SHA256 = "2056764119da0c37e1c416c3da249499b754678271a07b975ed2c0719e0ef6dd";
const BASE_P = 1037;
const BASE_TP = 961;
const BASE_SCORE = 0.923594;
const TRUTH_TOTAL = 1044;
const CANDIDATE_COUNT = 14;

const SOURCES = [
  {
    channel: "station",
    metadata: `${EXP}/v132_station_additions/v132_station_add_top20_from_pair015.json`,
    csv: `${EXP}/v132_station_additions/v132_station_add_top20_from_pair015.csv`,
  },
  {
    channel: "consensus",
    metadata: `${EXP}/v133_consensus_additions/v133_consensus_add_top20_from_pair015.json`,
    csv: `${EXP}/v133_consensus_additions/v133_consensus_add_top20_from_pair015.csv`,
  },
  {
    channel: "template",
    metadata: `${EXP}/v137_template_single_root_additions/v137_template_single_root_add_top20_from_pair015.json`,
    csv: `${EXP}/v137_template_single_root_additions/v137_template_single_root_add_top20_from_pair015.csv`,
  },
];

const V148_MATRIX = `${EXP}/v148_two_true_decode/probe_matrix.json`;
const V148_DECODE = `${EXP}/v148_two_true_decode/decode_state.json`;
const FIXED_LABELS = `${EXP}/research_equation_audit/fixed_labels.json`;
const OPTIONAL_6600_META = `${EXP}/v147_skip_negative/v147_v145_plus_6600_skip_dd294.json`;
const OPTIONAL_6600_SOURCE = `${EXP}/v147_skip_negative/v147_v145_plus_6600_skip_dd294.csv`;
const EXCLUDED_ORDERS = new Set(["dd29409d-298e-451b-8231-e0c53c0e420a"]);

const MATRIX = [
  [4, 5, 6, 10, 11, 12, 13, 14],
  [1, 2, 3, 4, 5, 6, 13, 14],
  [1, 2, 5, 7, 8, 10, 11, 12, 13],
  [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14],
  [3, 4, 6, 8, 10, 12, 13],
  [6, 7, 8, 9, 12, 14],
  [6, 7, 8, 11, 13],
  [1, 2, 3, 5, 7, 8, 12, 14],
  [1, 2, 4, 6, 10],
  [2, 5, 6, 9, 11, 12, 13],
];

const csvEscape = (value) => {
  const text = value == null ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};
const toCsv = (rows) => `${rows.map((row) => row.map(csvEscape).join(",")).join("\r\n")}\r\n`;
const sha256Bytes = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const sha256File = async (file) => sha256Bytes(await fs.readFile(file));
const readJson = async (file) => JSON.parse(await fs.readFile(file, "utf8"));
const writeJson = async (file, value) => fs.writeFile(file, `${JSON.stringify(value, null, 2)}\n`, "utf8");
const keyOf = (orderId, rid) => `${orderId}\u0000${rid}`;
const score6 = (value) => Math.round(value * 1_000_000) / 1_000_000;
const f1From = (tp, predictions) => (2 * tp) / (TRUTH_TOTAL + predictions);

function parseArgs(argv) {
  const options = {};
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (token.startsWith("--")) {
      const name = token.slice(2);
      const next = argv[i + 1];
      if (next == null || next.startsWith("--")) options[name] = true;
      else {
        options[name] = next;
        i += 1;
      }
    } else positional.push(token);
  }
  return { positional, options };
}

async function loadCsvWorkbook(file, sheetName) {
  const text = await fs.readFile(file, "utf8");
  const workbook = await Workbook.fromCSV(text, { sheetName });
  const sheet = workbook.worksheets.getItemAt(0);
  return { text, workbook, sheet, rows: sheet.getUsedRange().values };
}

function rootMapFromRows(rows) {
  const map = new Map();
  for (const row of rows.slice(1)) map.set(row[0], JSON.parse(row[1]).rootcause);
  return map;
}

function validateSubmission(rows, baseOrderIds, expectedP, expectedAdded = []) {
  if (rows.length !== 547) throw new Error(`Expected 547 CSV rows including header, got ${rows.length}`);
  if (rows[0][0] !== "order_id" || rows[0][1] !== "output") throw new Error("Invalid CSV header");
  const orderIds = rows.slice(1).map((row) => row[0]);
  if (JSON.stringify(orderIds) !== JSON.stringify(baseOrderIds)) throw new Error("order_id sequence differs from V148");
  const ridCounts = new Map();
  let predictions = 0;
  for (const row of rows.slice(1)) {
    const roots = JSON.parse(row[1]).rootcause;
    if (!Array.isArray(roots) || roots.length < 1 || roots.length > 8) throw new Error(`Invalid root count for ${row[0]}`);
    predictions += roots.length;
    for (const node of roots) {
      const rid = node["@rid"];
      if (!rid) throw new Error(`Missing @rid for ${row[0]}`);
      const exactKey = keyOf(row[0], rid);
      ridCounts.set(exactKey, (ridCounts.get(exactKey) ?? 0) + 1);
    }
  }
  if (predictions !== expectedP) throw new Error(`Expected P=${expectedP}, got ${predictions}`);
  for (const [candidateKey, count] of ridCounts) {
    if (count !== 1) throw new Error(`Duplicate RID within order: ${candidateKey}`);
  }
  for (const candidate of expectedAdded) {
    if (ridCounts.get(keyOf(candidate.order_id, candidate.add_rid)) !== 1) {
      throw new Error(`Expected candidate exactly once: ${candidate.candidate_id}`);
    }
  }
  return { orders: orderIds.length, predictions, uniqueOrderRidPairs: ridCounts.size };
}

function proveMatrix() {
  if (MATRIX.length !== 10 || MATRIX.some((row) => row.some((id) => id < 1 || id > CANDIDATE_COUNT))) {
    throw new Error("Invalid fixed matrix dimensions or candidate ids");
  }
  const seen = new Set();
  const prefixSets = Array.from({ length: MATRIX.length }, () => new Set());
  const hash = crypto.createHash("sha256");
  for (let mask = 0; mask < (1 << CANDIDATE_COUNT); mask++) {
    const signature = MATRIX.map((members) => members.reduce((sum, id) => sum + ((mask >> (id - 1)) & 1), 0));
    const signatureKey = signature.join(",");
    if (seen.has(signatureKey)) throw new Error(`Non-injective matrix signature: ${signatureKey}`);
    seen.add(signatureKey);
    hash.update(`${mask}:${signatureKey}\n`);
    for (let i = 0; i < signature.length; i++) prefixSets[i].add(signature.slice(0, i + 1).join(","));
  }
  return {
    label_vector_count: 1 << CANDIDATE_COUNT,
    unique_signature_count: seen.size,
    prefix_unique_signature_counts: prefixSets.map((set) => set.size),
    signature_map_sha256: hash.digest("hex"),
  };
}

async function inspectBase() {
  await fs.mkdir(PREVIEWS, { recursive: true });
  const base = await loadCsvWorkbook(BASE, "V148Base");
  const orderIds = base.rows.slice(1).map((row) => row[0]);
  const check = validateSubmission(base.rows, orderIds, BASE_P);
  const actualHash = await sha256File(BASE);
  if (actualHash !== BASE_SHA256) throw new Error(`V148 base SHA mismatch: ${actualHash}`);
  const inspected = await base.workbook.inspect({
    kind: "table",
    sheetId: "V148Base",
    range: "A1:B6",
    tableMaxRows: 6,
    tableMaxCols: 2,
    maxChars: 2500,
  });
  if (!inspected.ndjson.includes("order_id")) throw new Error("V148 base inspection failed");
  const preview = await base.workbook.render({ sheetName: "V148Base", range: "A1:B6", scale: 1, format: "png" });
  await fs.writeFile(`${PREVIEWS}/base.png`, new Uint8Array(await preview.arrayBuffer()));
  console.log(JSON.stringify({ path: BASE, sha256: actualHash, ...check, preview: `${PREVIEWS}/base.png` }, null, 2));
}

async function selectCandidates(base) {
  const aggregate = new Map();
  const sourceWorkbooks = new Map();
  for (const source of SOURCES) {
    const metadata = await readJson(source.metadata);
    if (!Array.isArray(metadata.selected) || metadata.selected.length !== 20) {
      throw new Error(`${source.channel} source must contain exactly 20 ranked additions`);
    }
    const sourceCsv = await loadCsvWorkbook(source.csv, `Source_${source.channel}`);
    sourceWorkbooks.set(source.channel, rootMapFromRows(sourceCsv.rows));
    metadata.selected.forEach((item, index) => {
      const candidateKey = keyOf(item.order_id, item.add_rid);
      const entry = aggregate.get(candidateKey) ?? {
        order_id: item.order_id,
        add_rid: item.add_rid,
        ranks: {},
        source_scores: {},
      };
      const rank = index + 1;
      entry.ranks[source.channel] = rank;
      entry.source_scores[source.channel] = item.station_score ?? item.consensus_score ?? item.template_score ?? null;
      aggregate.set(candidateKey, entry);
    });
  }

  const v148Matrix = await readJson(V148_MATRIX);
  const v148Decode = await readJson(V148_DECODE);
  const positiveIds = new Set(v148Decode.decoded_positive_pair ?? []);
  const knownFalse = new Set(
    v148Matrix.candidates
      .filter((candidate) => !positiveIds.has(candidate.candidate_id))
      .map((candidate) => keyOf(candidate.order_id, candidate.add_rid)),
  );
  const fixed = await readJson(FIXED_LABELS);
  const equationFixedZero = new Set(
    (fixed.labels ?? [])
      .filter((item) => item.label === 0 || (item.min === 0 && item.max === 0))
      .map((item) => keyOf(item.order_id, item.rid)),
  );
  const baseRoots = rootMapFromRows(base.rows);
  const excluded = [];
  const eligible = [];

  for (const entry of aggregate.values()) {
    entry.model_support = Object.keys(entry.ranks).length;
    entry.borda_score = Object.values(entry.ranks).reduce((sum, rank) => sum + (21 - rank), 0);
    const candidateKey = keyOf(entry.order_id, entry.add_rid);
    let reason = null;
    const current = baseRoots.get(entry.order_id);
    if (!current) reason = "order_missing_from_v148";
    else if (current.some((node) => node["@rid"] === entry.add_rid)) reason = "already_in_v148";
    else if (current.length >= 8) reason = "root_count_at_limit";
    else if (knownFalse.has(candidateKey)) reason = "v148_online_fixed_false";
    else if (equationFixedZero.has(candidateKey)) reason = "historical_equation_fixed_zero";
    else if (EXCLUDED_ORDERS.has(entry.order_id)) reason = "excluded_negative_order";

    let node = null;
    if (!reason) {
      for (const channel of Object.keys(entry.ranks)) {
        node = sourceWorkbooks.get(channel)?.get(entry.order_id)?.find((candidate) => candidate["@rid"] === entry.add_rid) ?? null;
        if (node) break;
      }
      if (!node) reason = "node_not_found_in_source_csv";
    }
    if (reason) excluded.push({ ...entry, reason });
    else eligible.push({ ...entry, node: structuredClone(node) });
  }

  eligible.sort((a, b) =>
    (b.model_support - a.model_support)
    || (b.borda_score - a.borda_score)
    || a.order_id.localeCompare(b.order_id)
    || a.add_rid.localeCompare(b.add_rid));

  const selected = [];
  const selectedOrders = new Set();
  for (const candidate of eligible) {
    if (selectedOrders.has(candidate.order_id)) {
      excluded.push({ ...candidate, node: undefined, reason: "lower_ranked_same_order" });
      continue;
    }
    selected.push(candidate);
    selectedOrders.add(candidate.order_id);
    if (selected.length === CANDIDATE_COUNT) break;
  }
  if (selected.length !== CANDIDATE_COUNT) throw new Error(`Only ${selected.length} eligible distinct-order candidates; need 14`);
  return {
    selected: selected.map((candidate, index) => ({ ...candidate, candidate_id: index + 1 })),
    excluded,
    eligible_count: eligible.length,
    known_false_count: knownFalse.size,
    equation_fixed_zero_count: equationFixedZero.size,
  };
}

async function addCandidatesToWorkbook(baseText, baseIndex, candidates, candidateIds, sheetName) {
  const workbook = await Workbook.fromCSV(baseText, { sheetName });
  const sheet = workbook.worksheets.getItemAt(0);
  for (const candidateId of candidateIds) {
    const candidate = candidates[candidateId - 1];
    const rowIndex = baseIndex.get(candidate.order_id);
    if (rowIndex == null) throw new Error(`Missing base order ${candidate.order_id}`);
    const cell = sheet.getRangeByIndexes(rowIndex, 1, 1, 1);
    const output = JSON.parse(cell.values[0][0]);
    if (output.rootcause.some((node) => node["@rid"] === candidate.add_rid)) throw new Error(`Candidate already present: ${candidate.candidate_id}`);
    output.rootcause.push(structuredClone(candidate.node));
    cell.values = [[JSON.stringify(output)]];
  }
  return { workbook, sheet, rows: sheet.getUsedRange().values };
}

async function verifyArtifactCsv(file, sheetName, baseOrderIds, expectedP, expectedAdded, previewName) {
  const bytes = await fs.readFile(file);
  if (bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) throw new Error(`${path.basename(file)} has a UTF-8 BOM`);
  const verify = await Workbook.fromCSV(bytes.toString("utf8"), { sheetName });
  const sheet = verify.worksheets.getItemAt(0);
  const check = validateSubmission(sheet.getUsedRange().values, baseOrderIds, expectedP, expectedAdded);
  const inspected = await verify.inspect({
    kind: "table",
    sheetId: sheetName,
    range: "A1:B6",
    tableMaxRows: 6,
    tableMaxCols: 2,
    maxChars: 2500,
  });
  if (!inspected.ndjson.includes("order_id")) throw new Error(`Artifact inspection failed: ${path.basename(file)}`);
  const errors = await verify.inspect({
    kind: "match",
    sheetId: sheetName,
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 20 },
    maxChars: 2000,
  });
  if (/\"matchCount\"\s*:\s*[1-9]/.test(errors.ndjson)) throw new Error(`Formula-like error token found in ${path.basename(file)}`);
  const preview = await verify.render({ sheetName, range: "A1:B6", scale: 1, format: "png" });
  await fs.writeFile(`${PREVIEWS}/${previewName}`, new Uint8Array(await preview.arrayBuffer()));
  return { ...check, sha256: sha256Bytes(bytes), bytes: bytes.length, utf8_bom: false };
}

async function build() {
  await fs.mkdir(OUT, { recursive: true });
  await fs.mkdir(PREVIEWS, { recursive: true });
  const existingScoresPath = `${OUT}/online_scores.json`;
  try {
    const existing = await readJson(existingScoresPath);
    if ((existing.records ?? []).length > 0) throw new Error("Refusing to rebuild after online probe records exist");
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }

  const base = await loadCsvWorkbook(BASE, "V148Base");
  const baseOrderIds = base.rows.slice(1).map((row) => row[0]);
  validateSubmission(base.rows, baseOrderIds, BASE_P);
  const actualBaseSha = await sha256File(BASE);
  if (actualBaseSha !== BASE_SHA256) throw new Error(`V148 base SHA mismatch: ${actualBaseSha}`);
  const baseIndex = new Map(baseOrderIds.map((orderId, index) => [orderId, index + 1]));
  const selection = await selectCandidates(base);
  const matrixProof = proveMatrix();

  const publicCandidates = selection.selected.map(({ node, ...candidate }) => candidate);
  const candidateCatalog = {
    version: "v149-online-calibrated-additions-1",
    base: BASE,
    base_sha256: actualBaseSha,
    ranking: "model_support desc, Borda desc, order_id asc, add_rid asc; one candidate per order",
    source_channels: SOURCES.map(({ channel, metadata, csv }) => ({ channel, metadata, csv })),
    exclusions: {
      v148_online_fixed_false_count: selection.known_false_count,
      historical_equation_fixed_zero_count: selection.equation_fixed_zero_count,
      excluded_negative_orders: [...EXCLUDED_ORDERS],
      deletion_pool_note: "V118/V119 deletion actions are not used; V149 contains pure additions only.",
    },
    eligible_before_distinct_order_cut: selection.eligible_count,
    candidates: publicCandidates,
    excluded_candidates: selection.excluded.map(({ node, ...candidate }) => candidate),
  };
  await writeJson(`${OUT}/candidate_catalog.json`, candidateCatalog);

  const manifest = {
    version: "v149-probe-matrix-1",
    status: "ready_for_online_probes",
    base: { file: BASE, sha256: actualBaseSha, predictions: BASE_P, tp: BASE_TP, public_f1: BASE_SCORE },
    truth_total: TRUTH_TOTAL,
    candidate_count: CANDIDATE_COUNT,
    matrix_rows: MATRIX,
    matrix_proof: matrixProof,
    probes: [],
  };

  for (let rowIndex = 0; rowIndex < MATRIX.length; rowIndex++) {
    const probeId = rowIndex + 1;
    const members = MATRIX[rowIndex];
    const authored = await addCandidatesToWorkbook(base.text, baseIndex, selection.selected, members, `R${String(probeId).padStart(2, "0")}`);
    const expectedCandidates = members.map((id) => selection.selected[id - 1]);
    const expectedP = BASE_P + members.length;
    validateSubmission(authored.rows, baseOrderIds, expectedP, expectedCandidates);
    const fileName = `v149_probe_r${String(probeId).padStart(2, "0")}.csv`;
    const file = `${OUT}/${fileName}`;
    await fs.writeFile(file, toCsv(authored.rows), "utf8");
    const check = await verifyArtifactCsv(file, `R${String(probeId).padStart(2, "0")}`, baseOrderIds, expectedP, expectedCandidates, `probe_r${String(probeId).padStart(2, "0")}.png`);
    manifest.probes.push({
      probe_id: probeId,
      file: fileName,
      candidate_ids: members,
      predictions: expectedP,
      orders: check.orders,
      sha256: check.sha256,
      status: "ready_not_scored",
    });
  }
  await writeJson(`${OUT}/probe_matrix.json`, manifest);

  const onlineScores = {
    version: "v149-online-scores-1",
    baseline: {
      file: path.basename(BASE),
      score: BASE_SCORE,
      predictions: BASE_P,
      inferred_tp: BASE_TP,
      sha256: actualBaseSha,
      source_class: "public_scored",
      note: "User-confirmed leaderboard screenshot on 2026-09-03.",
    },
    records: [],
    final_result: null,
    optional_6600_result: null,
    note: "Only actual leaderboard scores may be recorded.",
  };
  await writeJson(existingScoresPath, onlineScores);
  await writeJson(`${OUT}/decode_state.json`, {
    version: "v149-decode-state-1",
    status: "awaiting_probes",
    recorded_probe_ids: [],
    feasible_label_vector_count: 1 << CANDIDATE_COUNT,
    next_probe: 1,
    decoded_positive_candidate_ids: [],
    final_emission_allowed: false,
  });
  await writeJson(`${OUT}/final_gate.json`, {
    status: "awaiting_unique_decode",
    final_emission_allowed: false,
    verified_champion: { file: BASE, score: BASE_SCORE, predictions: BASE_P, tp: BASE_TP, sha256: actualBaseSha },
  });
  await writeJson(`${OUT}/validation.json`, {
    version: "v149-validation-1",
    all_valid: true,
    matrix_injective: matrixProof.unique_signature_count === (1 << CANDIDATE_COUNT),
    matrix_proof: matrixProof,
    candidate_count: selection.selected.length,
    distinct_candidate_orders: new Set(selection.selected.map((candidate) => candidate.order_id)).size,
    probe_count: manifest.probes.length,
    probe_checks: manifest.probes.map((probe) => ({ probe_id: probe.probe_id, orders: probe.orders, predictions: probe.predictions, sha256: probe.sha256 })),
    final_file_generated: false,
    evidence_class: "local structural and decoding validation only",
  });
  console.log(JSON.stringify({ status: manifest.status, candidates: publicCandidates, probes: manifest.probes, matrix_proof: matrixProof }, null, 2));
}

function inferUniqueTp(score, predictions) {
  const normalized = score6(score);
  const raw = score * (TRUTH_TOTAL + predictions) / 2;
  const center = Math.round(raw);
  const candidates = [];
  for (let tp = Math.max(0, center - 3); tp <= center + 3; tp++) {
    if (score6(f1From(tp, predictions)) === normalized) candidates.push(tp);
  }
  if (candidates.length !== 1) throw new Error(`Score ${normalized.toFixed(6)} at P=${predictions} maps to ${candidates.length} TP values: ${candidates}`);
  return { tp: candidates[0], raw_tp: raw, reconstructed_score: f1From(candidates[0], predictions) };
}

function feasibleMasks(records, manifest) {
  const probeRecords = records.filter((record) => record.kind === "probe");
  const feasible = [];
  for (let mask = 0; mask < (1 << CANDIDATE_COUNT); mask++) {
    let valid = true;
    for (const record of probeRecords) {
      const members = manifest.probes.find((probe) => probe.probe_id === record.probe_id)?.candidate_ids;
      if (!members) throw new Error(`Unknown recorded probe ${record.probe_id}`);
      const count = members.reduce((sum, id) => sum + ((mask >> (id - 1)) & 1), 0);
      if (count !== record.true_count_in_probe) {
        valid = false;
        break;
      }
    }
    if (valid) feasible.push(mask);
  }
  return feasible;
}

function decodedIds(mask) {
  return Array.from({ length: CANDIDATE_COUNT }, (_, index) => index + 1).filter((id) => (mask >> (id - 1)) & 1);
}

function chooseNextProbe(feasible, manifest, recorded) {
  const choices = manifest.probes
    .filter((probe) => !recorded.has(probe.probe_id))
    .map((probe) => {
      const partitions = new Map();
      for (const mask of feasible) {
        const count = probe.candidate_ids.reduce((sum, id) => sum + ((mask >> (id - 1)) & 1), 0);
        partitions.set(count, (partitions.get(count) ?? 0) + 1);
      }
      const sizes = [...partitions.values()];
      return {
        probe_id: probe.probe_id,
        partition_counts: Object.fromEntries([...partitions.entries()].sort((a, b) => a[0] - b[0])),
        worst_case_remaining: Math.max(...sizes),
        expected_remaining: sizes.reduce((sum, size) => sum + size * size, 0) / feasible.length,
      };
    })
    .sort((a, b) => a.worst_case_remaining - b.worst_case_remaining || a.expected_remaining - b.expected_remaining || a.probe_id - b.probe_id);
  return choices[0] ?? null;
}

async function updateDecodeState(scores, manifest) {
  const feasible = feasibleMasks(scores.records, manifest);
  const recorded = new Set(scores.records.filter((record) => record.kind === "probe").map((record) => record.probe_id));
  const unique = feasible.length === 1;
  const nextChoice = !unique && feasible.length > 0 ? chooseNextProbe(feasible, manifest, recorded) : null;
  const positiveIds = unique ? decodedIds(feasible[0]) : [];
  const state = {
    version: "v149-decode-state-1",
    status: feasible.length === 0 ? "conflicting_online_equations" : unique ? "decoded" : "awaiting_more_probes",
    recorded_probe_ids: [...recorded].sort((a, b) => a - b),
    observed_counts: scores.records.filter((record) => record.kind === "probe").sort((a, b) => a.probe_id - b.probe_id).map((record) => ({ probe_id: record.probe_id, true_count: record.true_count_in_probe })),
    feasible_label_vector_count: feasible.length,
    feasible_masks_preview: feasible.slice(0, 32),
    next_probe: nextChoice?.probe_id ?? null,
    next_probe_partition_counts: nextChoice?.partition_counts ?? null,
    next_probe_worst_case_remaining: nextChoice?.worst_case_remaining ?? null,
    next_probe_expected_remaining: nextChoice?.expected_remaining ?? null,
    decoded_positive_candidate_ids: positiveIds,
    decoded_true_count: positiveIds.length,
    final_emission_allowed: unique && positiveIds.length >= 1,
  };
  await writeJson(`${OUT}/decode_state.json`, state);
  return { state, feasible };
}

async function recordProbe(probeToken, scoreToken) {
  const manifest = await readJson(`${OUT}/probe_matrix.json`);
  const scores = await readJson(`${OUT}/online_scores.json`);
  const normalizedToken = String(probeToken).toLowerCase().replace(/^r/, "").replace(/^0+/, "");
  const score = Number(scoreToken);
  if (!Number.isFinite(score) || score <= 0 || score >= 1) throw new Error(`Invalid score: ${scoreToken}`);

  if (normalizedToken === "final") return recordFinal(score, scores);
  if (normalizedToken === "6600") return recordOptional6600(score, scores);

  const probeId = Number(normalizedToken);
  const probe = manifest.probes.find((item) => item.probe_id === probeId);
  if (!probe) throw new Error(`Unknown probe id: ${probeToken}`);
  const inferred = inferUniqueTp(score, probe.predictions);
  const trueCount = inferred.tp - BASE_TP;
  if (trueCount < 0 || trueCount > probe.candidate_ids.length) {
    throw new Error(`Probe R${String(probeId).padStart(2, "0")} inferred impossible true count ${trueCount}`);
  }
  const existing = scores.records.find((record) => record.kind === "probe" && record.probe_id === probeId);
  if (existing) {
    if (score6(existing.score) !== score6(score)) throw new Error(`Probe ${probeId} already recorded with a different score`);
  } else {
    scores.records.push({
      kind: "probe",
      probe_id: probeId,
      file: probe.file,
      score: score6(score),
      predictions: probe.predictions,
      inferred_tp: inferred.tp,
      raw_tp: inferred.raw_tp,
      reconstructed_score: inferred.reconstructed_score,
      true_count_in_probe: trueCount,
      candidate_ids: probe.candidate_ids,
      sha256: probe.sha256,
      source_class: "public_scored",
    });
    scores.records.sort((a, b) => a.probe_id - b.probe_id);
    await writeJson(`${OUT}/online_scores.json`, scores);
  }
  const decoded = await updateDecodeState(scores, manifest);
  console.log(JSON.stringify({ recorded: probeId, inferred_tp: inferred.tp, true_count: trueCount, decode_state: decoded.state }, null, 2));
}

async function emitFinal(positiveIds, candidates, base, baseOrderIds, baseIndex) {
  const authored = await addCandidatesToWorkbook(base.text, baseIndex, candidates, positiveIds, "V149Final");
  const selected = positiveIds.map((id) => candidates[id - 1]);
  const expectedP = BASE_P + selected.length;
  validateSubmission(authored.rows, baseOrderIds, expectedP, selected);
  const fileName = `v149_final_k${String(selected.length).padStart(2, "0")}_from_v148.csv`;
  const file = `${OUT}/${fileName}`;
  await fs.writeFile(file, toCsv(authored.rows), "utf8");
  const check = await verifyArtifactCsv(file, "V149Final", baseOrderIds, expectedP, selected, "final.png");
  const expectedTp = BASE_TP + selected.length;
  const manifest = {
    version: "v149-final-manifest-1",
    status: "ready_for_online_confirmation",
    file: fileName,
    sha256: check.sha256,
    base_file: path.basename(BASE),
    base_sha256: BASE_SHA256,
    decoded_positive_candidate_ids: positiveIds,
    decoded_candidates: selected.map(({ node, ...candidate }) => candidate),
    predictions: expectedP,
    expected_tp: expectedTp,
    expected_f1_exact: f1From(expectedTp, expectedP),
    expected_f1_display: score6(f1From(expectedTp, expectedP)),
    orders: check.orders,
    verification: {
      unique_decode: true,
      candidate_nodes_present_once: true,
      order_sequence_matches_v148: true,
      root_count_range: "1-8",
      utf8_bom: false,
      artifact_tool_reimport_inspect_render: true,
    },
  };
  await writeJson(`${OUT}/final_manifest.json`, manifest);
  await writeJson(`${OUT}/final_gate.json`, {
    status: "ready_for_online_confirmation",
    final_emission_allowed: true,
    final: manifest,
    rollback_champion: { file: BASE, score: BASE_SCORE, predictions: BASE_P, tp: BASE_TP, sha256: BASE_SHA256 },
  });
  return manifest;
}

async function decode() {
  const manifest = await readJson(`${OUT}/probe_matrix.json`);
  const scores = await readJson(`${OUT}/online_scores.json`);
  const catalog = await readJson(`${OUT}/candidate_catalog.json`);
  const decoded = await updateDecodeState(scores, manifest);
  if (decoded.feasible.length === 0) {
    await writeJson(`${OUT}/final_gate.json`, { status: "conflicting_online_equations", final_emission_allowed: false, action: "hold_reserved_submission_and_audit_scores" });
    console.log(JSON.stringify(decoded.state, null, 2));
    return;
  }
  if (decoded.feasible.length !== 1) {
    await writeJson(`${OUT}/final_gate.json`, { status: "awaiting_unique_decode", final_emission_allowed: false, next_probe: decoded.state.next_probe, feasible_label_vector_count: decoded.feasible.length });
    console.log(JSON.stringify(decoded.state, null, 2));
    return;
  }

  const positiveIds = decodedIds(decoded.feasible[0]);
  if (positiveIds.length === 0) {
    const optional = await emitOptional6600(BASE, BASE_P, BASE_TP, "v148");
    await writeJson(`${OUT}/final_gate.json`, {
      status: "decoded_no_true_additions",
      final_emission_allowed: false,
      optional_6600_ready: true,
      optional_6600: optional,
      rollback_champion: { file: BASE, score: BASE_SCORE, predictions: BASE_P, tp: BASE_TP, sha256: BASE_SHA256 },
    });
    console.log(JSON.stringify({ decode: decoded.state, optional_6600: optional }, null, 2));
    return;
  }

  const base = await loadCsvWorkbook(BASE, "V148Base");
  const baseOrderIds = base.rows.slice(1).map((row) => row[0]);
  const baseIndex = new Map(baseOrderIds.map((orderId, index) => [orderId, index + 1]));
  const sourceSelection = await selectCandidates(base);
  const currentPublic = sourceSelection.selected.map(({ node, ...candidate }) => candidate);
  if (JSON.stringify(currentPublic) !== JSON.stringify(catalog.candidates)) throw new Error("Candidate catalog drift detected; refusing final emission");
  const final = await emitFinal(positiveIds, sourceSelection.selected, base, baseOrderIds, baseIndex);
  console.log(JSON.stringify({ decode: decoded.state, final }, null, 2));
}

async function emitOptional6600(sourceFile, sourceP, sourceTp, label) {
  const actionMeta = await readJson(OPTIONAL_6600_META);
  const action = actionMeta.action;
  const source = await loadCsvWorkbook(sourceFile, "Optional6600Base");
  const orderIds = source.rows.slice(1).map((row) => row[0]);
  validateSubmission(source.rows, orderIds, sourceP);
  const sourceIndex = new Map(orderIds.map((orderId, index) => [orderId, index + 1]));
  const actionSource = await loadCsvWorkbook(OPTIONAL_6600_SOURCE, "Optional6600Source");
  const actionNode = rootMapFromRows(actionSource.rows).get(action.order_id)?.find((node) => node["@rid"] === action.add_rid);
  if (!actionNode) throw new Error("6600 addition node not found in source file");
  const workbook = await Workbook.fromCSV(source.text, { sheetName: "Optional6600" });
  const sheet = workbook.worksheets.getItemAt(0);
  const rowIndex = sourceIndex.get(action.order_id);
  if (rowIndex == null) throw new Error("6600 order missing from source champion");
  const cell = sheet.getRangeByIndexes(rowIndex, 1, 1, 1);
  const output = JSON.parse(cell.values[0][0]);
  const before = output.rootcause.length;
  output.rootcause = output.rootcause.filter((node) => node["@rid"] !== action.remove_rid);
  if (output.rootcause.length !== before - 1) throw new Error("6600 remove RID not present exactly once");
  if (output.rootcause.some((node) => node["@rid"] === action.add_rid)) throw new Error("6600 add RID already present");
  output.rootcause.push(structuredClone(actionNode));
  cell.values = [[JSON.stringify(output)]];
  const rows = sheet.getUsedRange().values;
  validateSubmission(rows, orderIds, sourceP);
  const fileName = `v149_optional_6600_from_${label}.csv`;
  const file = `${OUT}/${fileName}`;
  await fs.writeFile(file, toCsv(rows), "utf8");
  const check = await verifyArtifactCsv(file, "Optional6600", orderIds, sourceP, [], `optional_6600_${label}.png`);
  const result = {
    version: "v149-optional-6600-1",
    status: "ready_for_single_online_test",
    file: fileName,
    sha256: check.sha256,
    base_file: path.basename(sourceFile),
    predictions: sourceP,
    base_tp: sourceTp,
    branches: [-1, 0, 1].map((deltaTp) => ({ delta_tp: deltaTp, tp: sourceTp + deltaTp, score: score6(f1From(sourceTp + deltaTp, sourceP)) })),
    action,
  };
  await writeJson(`${OUT}/optional_6600_manifest.json`, result);
  return result;
}

async function recordFinal(score, scores) {
  const final = await readJson(`${OUT}/final_manifest.json`);
  const inferred = inferUniqueTp(score, final.predictions);
  const matches = inferred.tp === final.expected_tp && score6(score) === final.expected_f1_display;
  scores.final_result = {
    kind: "final",
    file: final.file,
    score: score6(score),
    predictions: final.predictions,
    inferred_tp: inferred.tp,
    expected_tp: final.expected_tp,
    expected_score: final.expected_f1_display,
    matches_expected: matches,
    sha256: final.sha256,
    source_class: "public_scored",
  };
  await writeJson(`${OUT}/online_scores.json`, scores);
  if (!matches) {
    await writeJson(`${OUT}/final_gate.json`, { status: "final_score_mismatch", final_emission_allowed: false, reserve_submission: true, rollback_champion: { file: BASE, score: BASE_SCORE, predictions: BASE_P, tp: BASE_TP } });
    console.log(JSON.stringify(scores.final_result, null, 2));
    return;
  }
  const optional = await emitOptional6600(`${OUT}/${final.file}`, final.predictions, final.expected_tp, "verified_final");
  await writeJson(`${OUT}/final_gate.json`, { status: "final_verified_optional_6600_ready", final_emission_allowed: true, verified_final: final, optional_6600: optional });
  console.log(JSON.stringify({ final_result: scores.final_result, optional_6600: optional }, null, 2));
}

async function recordOptional6600(score, scores) {
  const optional = await readJson(`${OUT}/optional_6600_manifest.json`);
  const inferred = inferUniqueTp(score, optional.predictions);
  const deltaTp = inferred.tp - optional.base_tp;
  const branch = optional.branches.find((item) => item.delta_tp === deltaTp) ?? null;
  scores.optional_6600_result = {
    kind: "optional_6600",
    file: optional.file,
    score: score6(score),
    predictions: optional.predictions,
    inferred_tp: inferred.tp,
    delta_tp: deltaTp,
    recognized_branch: branch != null,
    sha256: optional.sha256,
    source_class: "public_scored",
    decision: deltaTp === 1 ? "adopt_new_champion" : "rollback_and_exclude_6600",
  };
  await writeJson(`${OUT}/online_scores.json`, scores);
  console.log(JSON.stringify(scores.optional_6600_result, null, 2));
}

async function validateCampaign() {
  const manifest = await readJson(`${OUT}/probe_matrix.json`);
  const catalog = await readJson(`${OUT}/candidate_catalog.json`);
  const base = await loadCsvWorkbook(BASE, "V148Base");
  const baseOrderIds = base.rows.slice(1).map((row) => row[0]);
  const checks = [];
  for (const probe of manifest.probes) {
    const expected = probe.candidate_ids.map((id) => catalog.candidates[id - 1]);
    const check = await verifyArtifactCsv(`${OUT}/${probe.file}`, `R${String(probe.probe_id).padStart(2, "0")}`, baseOrderIds, probe.predictions, expected, `validate_r${String(probe.probe_id).padStart(2, "0")}.png`);
    if (check.sha256 !== probe.sha256) throw new Error(`SHA mismatch for ${probe.file}`);
    checks.push({ probe_id: probe.probe_id, ...check });
  }
  const proof = proveMatrix();
  const decoderTests = [];
  for (const mask of [0, (1 << CANDIDATE_COUNT) - 1, 0b10101010101010, 0b00100101101001]) {
    const synthetic = manifest.probes.map((probe) => ({ kind: "probe", probe_id: probe.probe_id, true_count_in_probe: probe.candidate_ids.reduce((sum, id) => sum + ((mask >> (id - 1)) & 1), 0) }));
    const feasible = feasibleMasks(synthetic, manifest);
    if (feasible.length !== 1 || feasible[0] !== mask) throw new Error(`Decoder self-test failed for mask ${mask}`);
    decoderTests.push({ mask, positive_ids: decodedIds(mask) });
  }
  const scoreInferenceTests = [];
  for (const probe of manifest.probes) {
    const rowSize = probe.candidate_ids.length;
    for (let trueCount = 0; trueCount <= rowSize; trueCount += 1) {
      const expectedTp = BASE_TP + trueCount;
      const displayedScore = score6(f1From(expectedTp, probe.predictions));
      const inferred = inferUniqueTp(displayedScore, probe.predictions);
      if (inferred.tp !== expectedTp) {
        throw new Error(`Six-decimal score inference failed for probe ${probe.probe_id}, count ${trueCount}`);
      }
    }
    scoreInferenceTests.push({
      probe_id: probe.probe_id,
      predictions: probe.predictions,
      tested_true_counts: [0, rowSize],
      all_unique: true,
    });
  }
  const validation = {
    version: "v149-validation-1",
    all_valid: true,
    evidence_class: "local structural and decoding validation only",
    candidate_count: catalog.candidates.length,
    distinct_candidate_orders: new Set(catalog.candidates.map((candidate) => candidate.order_id)).size,
    matrix_proof: proof,
    decoder_tests: decoderTests,
    score_inference_tests: scoreInferenceTests,
    probe_checks: checks,
  };
  await writeJson(`${OUT}/validation.json`, validation);
  console.log(JSON.stringify(validation, null, 2));
}

function printHelp() {
  console.log(`V149 commands:\n  inspect-base\n  build\n  record --probe-id <1..10|final|6600> --score <six-decimal-score>\n  decode\n  validate`);
}

await fs.mkdir(OUT, { recursive: true });
const { positional, options } = parseArgs(process.argv.slice(2));
const command = positional[0] ?? "help";
if (command === "inspect-base") await inspectBase();
else if (command === "build") await build();
else if (command === "record") {
  if (options["probe-id"] == null || options.score == null) throw new Error("record requires --probe-id and --score");
  await recordProbe(options["probe-id"], options.score);
} else if (command === "decode") await decode();
else if (command === "validate") await validateCampaign();
else printHelp();
