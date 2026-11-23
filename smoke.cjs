// smoke.cjs — 页面守卫两件套：
//   1) 静态：CSS 存在性 / 结构 / 文案口径 / 锚点常量
//   2) 运行时：自动桩 DOM 下真跑页面脚本 + setTab('mem') + renderMem 双分支，
//      抓渲染作用域错误（如引用了不在作用域的标识符导致整个 renderMem 中断）
// 数值动态校验（652 项，网页模型 vs python/头文件公式）见 verify_model.mjs：
//   node verify_model.mjs ref/r.csv ref/e.csv ref/layout_ref.json
// 用法：node smoke.cjs [index.html 路径]
const { readFileSync } = require("fs");
const path = require("path");
const file = process.argv[2] || path.join(__dirname, "index.html");
const html = readFileSync(file, "utf8");
let fails = 0, checks = 0;
const ok = (cond, msg) => { checks++; if (!cond) { fails++; console.error("FAIL: " + msg); } };

// ---------- 1) 静态守卫 ----------
for (const [nm, pat] of [["CSS tabbar", ".tabbar{"], ["CSS memcfg", ".memcfg .grid{"],
  ["CSS ltg", ".ltg{"], ["CSS detcard", ".detcard summary{"],
  ["CSS kbar", ".kbar{"], ["CSS kbrk", ".kbrk{"]])
  ok(html.includes(pat), nm);
for (const id of ["tabRoofline", "tabMemory", "tabMfu", "memKpis", "memTbl", "memDecide", "mm-csv",
  "mm-anchor", "m1", "m2", "m3", "m4", "ch1", "in-m", "mm-hbm", "mm-f", "mm-r",
  "mfuForm", "in-mfu-ms", "in-mfu-base", "in-mfu-ep", "in-mfu-f", "mfuKpis", "mfuRoofKpis",
  "tabPipe", "pipeFwdTbl", "pipeBwdTbl", "pipeTraceTbl", "segFwdSched", "segBwdSave",
  "ganttCard", "ganttFile", "ganttDemo", "ganttHost", "ganttEmpty", "ganttLegend", "ganttMeta",
  "pipeSchematic", "schedHost", "schedLegend", "schedCaption", "segSched", "segWaves"])
  ok(html.includes(`id="${id}"`), `容器 id=${id}`);
ok(html.includes("实测 MFU"), "MFU 卡片标题");
ok(html.includes("Wave 循环，五阶段") && html.includes("按专家 128 行块流水"), "流水页两节标题");
ok(html.includes(".pstage{") && html.includes(".parrow{"), "流水页样式");
ok(html.includes("wave_7_AIC_BW_DW2_EARLY_E1") && html.includes("AIV_BW_DP_REDUCE_SEND"), "事件命名对照");
ok(html.includes("甘特图：导入 timer CSV"), "甘特图区块");
ok(html.includes("流水示意") && html.includes("function buildFwdSchematic"), "流水示意区块");
ok(html.includes("SCHED_MEAS") && html.includes("波周期 228") && html.includes("不是 2×"), "示意以实测为据");
ok(html.includes("dW1 ≈ 1434") && html.includes("INIT_ROUTING ≈ 90"), "反向尾部/固定开销实测标注");
ok(html.includes("function parseTimerCsv") && html.includes("_start_cycle"), "CSV 解析器");
ok(html.includes("segMfuMode") && html.includes("segMfuSave"), "MFU 阶段 × gate/up 双维度选择");
ok(html.includes("保存 · 免重算") && html.includes("重算 GMM1"), "MFU 反向 12/16RFK 可选");
ok(html.includes("前向显存峰值 / rank"), "KPI 前向峰值卡");
ok(html.includes("反向显存峰值 / rank"), "KPI 反向峰值卡");
ok(html.includes("SHMEM 额度（= local_mem_size）"), "KPI SHMEM 额度卡");
ok(html.includes("Align2MiB(布局需要量)"), "SHMEM 按需口径文案");
ok(html.includes("前向保存 → 反向使用（saved + dispatch 输出）"), "KPI saved 卡（含 dispatch 输出）");
ok(html.includes("dispatch 输出") && html.includes("重派发"), "dispatch 输出口径");
ok(html.includes("retain_dispatch_output"), "retain 开关口径");
ok(html.includes("kbar-seg") && html.includes("kbrk"), "KPI 分解条/图例");
ok(!html.includes("floor 富余"), "已删口径：floor 富余不回潮");
ok(!html.includes("可驻留 micro-batch（"), "已删口径：可驻留卡不回潮");
ok(html.includes("rank 同步头") && html.includes("72 KiB"), "SHMEM 同步头口径");
ok(html.includes("routeIndex 区") && html.includes("combine / topkGrad 数据区"), "SHMEM PeerPayload 行");
ok(html.includes("紧凑计数收件箱"), "EP≥512 紧凑收件箱口径");
ok(html.includes("R（maxRecvTokens 接收容量）"), "面板 R 符号对齐");
ok(html.includes("生命周期") && html.includes("F→B") && html.includes("仅 B") && html.includes("仅 F"), "决策表生命周期列");
ok(html.includes("xTitle: '训练阶段（每 rank）'"), "M1 xTitle");
ok(html.includes("peakMark: true") && html.includes("↑ 峰值"), "M1 峰值标注");
ok(html.includes("const BX = i => Lm + (i + 0.5) * iw / n"), "柱状图槽位居中");
for (const v of ["529408", "1074271232", "554177536", "939655168", "2013926400",
  "3628074048", "2550441536", "536871168", "998846464"])
  ok(html.includes(v), `锚点常量 ${v}`);
ok(html.includes("/*MODEL-START*/") && html.includes("/*MODEL-END*/"), "MODEL 标记");
ok(html.includes("/*MEM-MODEL-START*/") && html.includes("/*MEM-MODEL-END*/"), "MEM-MODEL 标记");

// ---------- 2) 运行时渲染守卫 ----------
function makeEl(tag, value) {
  return {
    tag, children: [], style: {}, dataset: {}, value: value || "", _text: "",
    classList: { _s: new Set(),
      add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
      toggle(c, f) { if (f === undefined) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); } else if (f) this._s.add(c); else this._s.delete(c); },
      contains(c) { return this._s.has(c); } },
    setAttribute(k, v) { this["@" + k] = String(v); },
    getAttribute() { return null; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {}, removeEventListener() {}, click() {}, focus() {},
    getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100 }; },
    get textContent() { return this._text; }, set textContent(v) { this._text = String(v); this.children = []; },
    get innerHTML() { return this._text; }, set innerHTML(v) { this._text = String(v); this.children = []; },
    get firstChild() { return this.children[0] || null; },
  };
}
const els = {};
const getId = id => els[id] || (els[id] = makeEl("div"));
// 从 markup 种入 input 初值（init/URL hash 为空时的缺省口径）
for (const m of html.matchAll(/<input[^>]*>/g)) {
  const tag = m[0], id = /id="([\w-]+)"/.exec(tag), val = /value="([^"]*)"/.exec(tag);
  if (id) els[id[1]] = makeEl("input", val ? val[1] : "");
}
// 分段按钮组：#segVar/.active 等 — 每组一个可变 dataset 的桩
const groupActive = {};
for (const m of html.matchAll(/id="(seg\w+)"/g)) {
  const gid = m[1];
  const blk = html.slice(m.index, m.index + 2000);
  const act = /data-v="([^"]+)"[^>]*class="active"/.exec(blk) || /class="active"[^>]*data-v="([^"]+)"/.exec(blk);
  groupActive[gid] = makeEl("button");
  groupActive[gid].dataset.v = act ? act[1] : "";
}
function qs(sel) {
  const m = /^#(seg\w+) \.active$/.exec(sel);
  if (m) return groupActive[m[1]];
  const id = /^#([\w-]+)$/.exec(sel);
  return id ? getId(id[1]) : makeEl("div");
}
global.document = {
  getElementById: getId, querySelector: qs, querySelectorAll: () => [],
  createElement: t => makeEl(t), createElementNS: (ns, t) => makeEl(t),
  body: makeEl("body"), documentElement: makeEl("html"), addEventListener() {},
};
global.window = global;
global.location = { hash: "", search: "" };
global.history = { replaceState() {} };
global.localStorage = { getItem: () => null, setItem() {} };
global.addEventListener = () => {};

const code = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join("\n") +
  "\n;globalThis.__smoke = (function(){" +
  " setTab('mem');" +
  " const kpis = document.getElementById('memKpis').innerHTML;" +
  " const m1n = document.getElementById('m1').children.length;" +
  " const m3n = document.getElementById('m3').children.length;" +
  " const tbl = document.getElementById('memTbl').innerHTML;" +
  " document.querySelector('#segVar .active').dataset.v = 'udma'; setTab('mem');" +
  " const kpisU = document.getElementById('memKpis').innerHTML;" +
  " const m1nU = document.getElementById('m1').children.length;" +
  " setTab('pipe');" +
  " const pipeTabShown = document.getElementById('tabPipe').style.display === '';" +
  " const fwdTbl = document.getElementById('pipeFwdTbl').innerHTML;" +
  " const bwdTbl = document.getElementById('pipeBwdTbl').innerHTML;" +
  " const traceTbl = document.getElementById('pipeTraceTbl').innerHTML;" +
  " document.querySelector('#segFwdSched .active').dataset.v = '1'; renderPipe();" +
  " const fwdNote1 = document.getElementById('pipeFwdNote').innerHTML;" +
  " setTab('pipe'); loadGanttCsv(ganttDemoCsv(), '示例数据');" +
  " const ganttSvg = document.getElementById('ganttHost').children.length;" +
  " const ganttMetaTxt = document.getElementById('ganttMeta').textContent;" +
  " const ganttLeg = document.getElementById('ganttLegend').innerHTML;" +
  " const schedSvg = document.getElementById('schedHost').children.length;" +
  " const schedLeg = document.getElementById('schedLegend').innerHTML;" +
  " document.querySelector('#segSched .active').dataset.v = 'bwd'; renderPipe();" +
  " const schedBwd = document.getElementById('schedCaption').textContent;" +
  " setTab('mfu');" +
  " const m4n = document.getElementById('m4').children.length;" +
  " const mfuTabShown = document.getElementById('tabMfu').style.display === '';" +
  " const mfuK = document.getElementById('mfuKpis').innerHTML;" +
  " return { kpis, m1n, m3n, tbl, kpisU, m1nU, m4n, mfuTabShown, mfuK };" +
  "})();";
// MFU 运行时：init 已随 render() 渲染；再注入实测耗时验证数值路径
const code2 = code.replace("return { kpis, m1n, m3n, tbl, kpisU, m1nU, m4n, mfuTabShown, mfuK };",
  " const mfu = document.getElementById('mfuKpis').innerHTML;" +
   " document.getElementById('in-mfu-ms').value = '100';" +
  " document.getElementById('in-mfu-base').value = '280'; renderMfu();" +
  " const mfu2 = document.getElementById('mfuKpis').innerHTML;" +
  " const layerSave = document.getElementById('mfuCoefNote').textContent;" +
  " document.querySelector('#segMfuMode .active').dataset.v = 'fwd'; renderMfu();" +
  " const mfuF = document.getElementById('mfuKpis').innerHTML;" +
  " const mfuR = document.getElementById('mfuRoofKpis').innerHTML;" +
  " document.querySelector('#segMfuMode .active').dataset.v = 'bwd';" +
  " document.querySelector('#segMfuSave .active').dataset.v = 'save'; renderMfu();" +
  " const bwdSave = document.getElementById('mfuCoefNote').textContent;" +
  " document.querySelector('#segMfuSave .active').dataset.v = 'recompute'; renderMfu();" +
  " const bwdRec = document.getElementById('mfuCoefNote').textContent;" +
  " return { kpis, m1n, m3n, tbl, kpisU, m1nU, m4n, mfuTabShown, mfuK, mfu, mfu2, mfuF, mfuR, layerSave, bwdSave, bwdRec, pipeTabShown, fwdTbl, bwdTbl, traceTbl, fwdNote1, ganttSvg, ganttMetaTxt, ganttLeg, schedSvg, schedLeg, schedBwd };");
let P = null, runErr = null;
try { P = new Function(code2)() || globalThis.__smoke; } catch (e) { runErr = e; }

if (runErr) { checks++; fails++;
  console.error("FAIL: 页面脚本运行中断 — " + runErr.message);
  if (runErr.stack) console.error(runErr.stack.split("\n").slice(1, 4).join("\n")); }
else {
  ok((P.kpis || "").includes("前向显存峰值 / rank"), "运行时：normal 分支 KPI 渲染");
  ok(P.m1n === 1, "运行时：M1 SVG 渲染 (" + P.m1n + ")");
  ok(P.m3n === 1, "运行时：M3 SVG 渲染 (" + P.m3n + ")");
  ok((P.kpis || "").includes("rank 同步头") && (P.kpis || "").includes("combine / topkGrad 数据区"), "运行时：SHMEM 构成条");
  ok((P.tbl || "").includes("rank 同步头"), "运行时：账本同步头行");
  ok((P.kpisU || "").includes("PeerPayload") || (P.kpisU || "").includes("rank 同步头"), "运行时：UDMA 变体渲染");
  ok((P.mfu || "").includes("计算地板 C") && !(P.mfu || "").includes("通信 N"), "运行时：MFU 核心卡（无通信依赖）");
  ok((P.mfuR || "").includes("通信 N") && (P.mfuR || "").includes("融合上限"), "运行时：roofline 对比卡（折叠区）");
  ok((P.mfu2 || "").includes("%") && (P.mfu2 || "").includes("TFLOPS"), "运行时：MFU 数值计算");
  ok(P.m4n === 1 && P.mfuTabShown === true, "运行时：MFU 独立页渲染");
  ok((P.mfuK || "").includes("计算地板 C"), "运行时：MFU 页 KPI");
  ok(P.pipeTabShown === true && (P.fwdTbl || "").includes("模式 1"), "运行时：流水页前向表");
  ok((P.bwdTbl || "").includes("重算 GMM1") && (P.traceTbl || "").includes("EARLY"), "运行时：流水页反向/事件表");
  ok((P.fwdNote1 || "").includes("分阶段"), "运行时：流水页模式切换生效");
  ok(P.ganttSvg === 1, "运行时：甘特图渲染 SVG (" + P.ganttSvg + ")");
  ok((P.ganttMetaTxt || "").includes("lane") && (P.ganttMetaTxt || "").includes("事件"), "运行时：甘特图统计行");
  ok((P.ganttLeg || "").includes("AIC_BW_DHP"), "运行时：甘特图图例");
  ok(P.schedSvg === 1 && (P.schedLeg || "").includes("AIC_GMM1"), "运行时：流水示意图渲染");
  ok((P.schedBwd || "").includes("反向") && (P.schedBwd || "").includes("dO"), "运行时：示意切反向");
  ok((P.mfuF || "").includes("6·R·F·K"), "运行时：前向口径切换（6RFK）");
ok((P.layerSave || "").includes("coef = 18RFK"), "运行时：前向+反向+保存 = 18RFK");
ok((P.bwdSave || "").includes("coef = 12RFK"), "运行时：反向+保存 = 12RFK");
ok((P.bwdRec || "").includes("coef = 16RFK"), "运行时：反向+重算 = 16RFK");
  ok(P.m1nU === 1, "运行时：UDMA M1 渲染 (" + P.m1nU + ")");
}

const total = `SMOKE ${fails ? "FAIL" : "OK"} (${fails} 失败 / ${checks} 项)` +
  (fails ? "" : " — 数值动态校验: node verify_model.mjs ref/r.csv ref/e.csv ref/layout_ref.json");
console.log(total);
process.exit(fails ? 1 : 0);











