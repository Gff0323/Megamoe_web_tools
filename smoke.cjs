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
for (const id of ["tabRoofline", "tabMemory", "memKpis", "memTbl", "memDecide", "mm-csv",
  "mm-anchor", "m1", "m2", "m3", "ch1", "in-m", "mm-hbm", "mm-f", "mm-r",
  "mfuForm", "in-mfu-ms", "in-mfu-ep", "in-mfu-f", "mfuKpis"])
  ok(html.includes(`id="${id}"`), `容器 id=${id}`);
ok(html.includes("实测 MFU"), "MFU 卡片标题");
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
ok(html.includes("同步/控制保留区") && html.includes("72 KiB"), "SHMEM 同步头口径");
ok(html.includes("紧凑计数收件箱"), "EP≥512 紧凑收件箱口径");
ok(html.includes("生命周期") && html.includes("F→B") && html.includes("仅 B") && html.includes("仅 F"), "决策表生命周期列");
ok(html.includes("xTitle: '训练阶段（每 rank）'"), "M1 xTitle");
ok(html.includes("peakMark: true") && html.includes("↑ 峰值"), "M1 峰值标注");
ok(html.includes("const BX = i => Lm + (i + 0.5) * iw / n"), "柱状图槽位居中");
for (const v of ["530432", "1074272256", "554178560", "939655168", "2013927424", "1879257088", "2818769920", "536936960"])
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
  " return { kpis, m1n, m3n, tbl, kpisU, m1nU };" +
  "})();";
// MFU 运行时：init 已随 render() 渲染；再注入实测耗时验证数值路径
const code2 = code.replace("return { kpis, m1n, m3n, tbl, kpisU, m1nU };",
  " const mfu = document.getElementById('mfuKpis').innerHTML;" +
  " document.getElementById('in-mfu-ms').value = '100'; renderMfu();" +
  " const mfu2 = document.getElementById('mfuKpis').innerHTML;" +
  " return { kpis, m1n, m3n, tbl, kpisU, m1nU, mfu, mfu2 };");
let P = null, runErr = null;
try { P = new Function(code2)() || globalThis.__smoke; } catch (e) { runErr = e; }

if (runErr) { checks++; fails++;
  console.error("FAIL: 页面脚本运行中断 — " + runErr.message);
  if (runErr.stack) console.error(runErr.stack.split("\n").slice(1, 4).join("\n")); }
else {
  ok((P.kpis || "").includes("前向显存峰值 / rank"), "运行时：normal 分支 KPI 渲染");
  ok(P.m1n === 1, "运行时：M1 SVG 渲染 (" + P.m1n + ")");
  ok(P.m3n === 1, "运行时：M3 SVG 渲染 (" + P.m3n + ")");
  ok((P.kpis || "").includes("同步/控制保留区") && (P.kpis || "").includes("发送槽"), "运行时：normal SHMEM 构成条");
  ok((P.tbl || "").includes("同步/控制保留区"), "运行时：账本同步头行");
  ok((P.kpisU || "").includes("ptrA1"), "运行时：UDMA 分支渲染");
  ok((P.mfu || "").includes("计算地板 C") && (P.mfu || "").includes("通信 N"), "运行时：MFU C/N 卡");
  ok((P.mfu2 || "").includes("%") && (P.mfu2 || "").includes("TFLOPS"), "运行时：MFU 数值计算");
  ok(P.m1nU === 1, "运行时：UDMA M1 渲染 (" + P.m1nU + ")");
}

const total = `SMOKE ${fails ? "FAIL" : "OK"} (${fails} 失败 / ${checks} 项)` +
  (fails ? "" : " — 数值动态校验: node verify_model.mjs ref/r.csv ref/e.csv ref/layout_ref.json");
console.log(total);
process.exit(fails ? 1 : 0);
