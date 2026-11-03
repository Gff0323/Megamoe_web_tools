// 校验网页模型与原 Python 脚本数值完全一致。
// 用法：
//   1) python test/bigop_fused/roofline_probe.py --csv /tmp/r.csv --out /tmp/r.png
//      python test/bigop_fused/ep_probe.py       --csv /tmp/e.csv --out /tmp/e.png
//      python test/bigop_fused/roofline_web/export_layout_reference.py > /tmp/layout_ref.json
//   2) node verify_model.mjs /tmp/r.csv /tmp/e.csv /tmp/layout_ref.json
// 输出 "ALL OK" 即逐单元格一致（含显存模型三变体与 docs 锚点）。
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
const m = html.match(/\/\*MODEL-START\*\/([\s\S]*?)\/\*MODEL-END\*\//);
if (!m){ console.error('MODEL block not found in index.html'); process.exit(1); }
const modelAnalyze = new Function(m[1] + '\nreturn modelAnalyze;')();

function parseCsv(path){
  const lines = readFileSync(path, 'utf8').split(/\r?\n/).filter(Boolean);
  const head = lines[0].split(',');
  return lines.slice(1).map(l => {
    const parts = l.split(',');
    const o = {};
    head.forEach((h, i) => o[h] = parts[i]);
    return o;
  });
}
let fails = 0, checks = 0;
function chk(name, web, py, isStr){
  checks++;
  let ok;
  if (isStr) ok = String(web) === String(py);
  else if ((py === 'inf' || py === 'Infinity') && web === null) ok = true;
  else {
    const x = parseFloat(web), y = parseFloat(py);
    ok = isFinite(x) && isFinite(y) && Math.abs(x - y) <= Math.max(Math.abs(x), Math.abs(y)) * 1e-9 + 1e-12;
  }
  if (!ok){ fails++; console.log(`  MISMATCH ${name}: web=${web} py=${py}`); }
}
function runSuite(label, pyRows, web, map){
  console.log(`${label}: ${pyRows.length} py rows vs ${web.rows.length} web rows`);
  if (pyRows.length !== web.rows.length){
    fails++; console.log('  ROW COUNT MISMATCH');
  }
  const n = Math.min(pyRows.length, web.rows.length);
  for (let i = 0; i < n; i++){
    for (const [pk, wk] of Object.entries(map)){
      const isStr = ['ep', 'f', 'nodes', 'racks', 'bottleneck', 'regime'].includes(pk);
      chk(`row${i}.${pk}`, web.rows[i][wk], pyRows[i][pk], isStr);
    }
  }
}

if (process.argv[2]){
  runSuite('roofline_probe (整层 22RFK, BW 247.8/141.6/141.6, EP→1024)',
    parseCsv(process.argv[2]),
    modelAnalyze({ m: 4096, k: 7168, topk: 8, fList: [3072, 4096, 8192], cube: 410.4,
      bwHccs: 247.8, bwRack: 141.6, bwCage: 141.6, rn: 8, rr: 8, coef: 22,
      eps: [8, 16, 32, 64, 128, 256, 512, 1024] }),
    { ep: 'ep', f: 'f', nodes: 'nodes', racks: 'racks', bottleneck: 'bottleneck',
      regime: 'regime', t_compute: 'tCompute', t_comm: 'tComm', t_way: 'tWay',
      comm_compute_ratio: 'ncRatio', t_smallop: 'tSmallop', t_ideal: 'tIdeal',
      speedup_ceil: 'speedup', overlap_pct: 'overlapPct' });
}
if (process.argv[3]){
  const web = modelAnalyze({ m: 4096, k: 7168, topk: 8, fList: [3072, 8192], cube: 410.4,
    bwHccs: 784, bwRack: 80, bwCage: 15, rn: 8, rr: 8, coef: 16,
    eps: [8, 16, 32, 64, 128, 256, 320] });
  // ep_probe 的 comm_ratio_pct = 100 * t_comm / t_bwd_compute，网页模型里即 ncRatio*100
  for (const r of web.rows) r.comm_ratio_pct = r.ncRatio === null ? null : r.ncRatio * 100;
  runSuite('ep_probe (仅反向 16RFK, BW 784/80/15, EP→320)',
    parseCsv(process.argv[3]), web,
    { ep: 'ep', f: 'f', nodes: 'nodes', racks: 'racks', bottleneck: 'bottleneck',
      mb_node: 'mbNode', mb_rack: 'mbRack', mb_cage: 'mbCage',
      t_node: 'tNode', t_rack: 'tRack', t_cage: 'tCage',
      t_way: 'tWay', t_layer: 'tComm', t_bwd_compute: 'tCompute',
      speedup_ceil: 'speedup', comm_ratio_pct: 'comm_ratio_pct' });
}
// 3) 显存模型 vs python test_workspace_layout.py + kernel 头文件公式 + docs 锚点
if (process.argv[4]){
  const memBlock = html.match(/\/\*MEM-MODEL-START\*\/([\s\S]*?)\/\*MEM-MODEL-END\*\//);
  if (!memBlock){ console.error('MEM-MODEL block not found in index.html'); process.exit(1); }
  const mem = new Function(memBlock[1] + '\nreturn { memModel, memAnchors };')();
  const ref = JSON.parse(readFileSync(process.argv[4], 'utf8'));
  console.log(`\n显存模型：${ref.length} 个 shape × 三变体 × gate/up 四档 vs python/头文件公式`);
  for (const s of ref){
    const mk = o => mem.memModel(Object.assign({}, s, o));
    const Nn = mk({ variant: 'normal', saveMode: 'none' });
    const Nb = mk({ variant: 'normal', saveMode: 'bf16' });
    const Nm = mk({ variant: 'normal', saveMode: 'mxfp8' });
    const Nc = mk({ variant: 'normal', saveMode: 'mxfp8', clipMasks: true });
    const G = mk({ variant: 'mgroup', saveMode: 'bf16' });
    const Gn = mk({ variant: 'mgroup', saveMode: 'none' });
    const U = mk({ variant: 'udma', saveMode: 'bf16' });
    const tag = `M${s.m}K${s.k}N${s.n}T${s.topk}EP${s.ep}E${s.epr}R${s.r}`;
    chk(`${tag} saved(none/legacy)`, Nn.savedT, s.saved_none);
    chk(`${tag} saved(bf16)`, Nb.savedT, s.saved_bf16);
    chk(`${tag} saved(mxfp8)`, Nm.savedT, s.saved_mxfp8);
    chk(`${tag} saved(mxfp8+clip)`, Nc.savedT, s.saved_mxfp8_clip);
    chk(`${tag} saved(mgroup)=嵌入档`, G.savedT, s.saved_mgroup_bf16);
    chk(`${tag} saved(udma)=bf16档(不含packedX)`, U.savedT, s.saved_bf16);
    chk(`${tag} dispatch 输出(normal)`, Nn.dispatchOutT, s.dispatch_output);
    chk(`${tag} dispatch 输出(udma 并入 SHMEM)`, U.dispatchOutT, 0);
    chk(`${tag} transient(normal)`, Nn.transT, s.transient_normal);
    chk(`${tag} transient(mgroup)`, Gn.transT, s.transient_mgroup);
    chk(`${tag} transient(udma)`, U.transT, s.transient_udma);
    chk(`${tag} backward(none含重算)`, Nn.bwdT, s.backward_none);
    chk(`${tag} backward(saved免重算)`, Nb.bwdT, s.backward_saved);
    chk(`${tag} shmem(normal)=PeermemInfo`, Nn.shmT, s.shmem_normal_cpp);
    const compact = s.ep >= 512 && s.epr > 0 && s.epr <= 8;
    const inbox = compact ? Math.ceil(s.ep * 64 / 512) * 512 + Math.ceil(72 * 4 / 512) * 512 : 0;
    chk(`${tag} shmem(normal)−routed−收件箱=py对称窗`,
      Nn.shmT - Math.ceil(2 * s.m * s.topk / 512) * 512 - inbox, s.shmem_symmetric_py);
    chk(`${tag} shmem(udma)`, U.shmT, s.shmem_udma);
  }
  console.log('docs 锚点（M4096/N8192/K7168/T16/EP8/EPR4/R65536，MXFP8 文档 + §3.3/§9.2）：');
  for (const [name, exp, got] of mem.memAnchors()) chk('anchor ' + name, got, exp);
}
console.log(fails ? `\n${fails}/${checks} 项不一致` : `\nALL OK (${checks} 项数值全部一致)`);
process.exit(fails ? 1 : 0);
