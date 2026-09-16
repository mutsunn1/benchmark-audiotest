/**
 * 中文口语评测基准 — 技术汇报 deck
 *
 * Palette: lab-instrument. Deep ink ground, amber signal accent (the motif
 * dot), brick for the failure numbers, moss for the baseline.
 */
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3" x 7.5"
pres.author = "captbench";
pres.title = "中文口语评测基准：传统 CAPT vs Qwen-Omni";

// ---------------------------------------------------------------- tokens
const INK = "16181D";
const SLATE = "2D3E50";
const AMBER = "E8A33D";
const BRICK = "B23A33";
const MOSS = "3F7D5A";
const PAPER = "F7F7F5";
const MUTED = "6B7280";
const LINE = "D8DCE0";
const WHITE = "FFFFFF";

const CJK = "Heiti SC"; // present in fc-list AND a stock macOS font
const NUM = "Cambria"; // safe-serif numerals, pairs against the sans CJK

const W = 13.333;
const H = 7.5;
const M = 0.75; // margin

// ---------------------------------------------------------------- helpers
/** Amber dot + label. The repeating motif: a signal point. */
function marker(slide, text, x, y, color) {
  slide.addShape(pres.ShapeType.ellipse, {
    x, y: y + 0.045, w: 0.13, h: 0.13,
    fill: { color: color || AMBER },
  });
  slide.addText(text, {
    x: x + 0.24, y, w: 8, h: 0.26,
    fontFace: CJK, fontSize: 12.5, color: MUTED, charSpacing: 1.6,
    margin: 0, valign: "middle",
  });
}

function title(slide, text, y, color, size) {
  slide.addText(text, {
    x: M, y, w: W - 2 * M, h: 0.8,
    fontFace: CJK, fontSize: size || 30, bold: true,
    color: color || SLATE, margin: 0, valign: "middle",
  });
}

/** Big measured number with a caption — the workhorse of this deck. */
function stat(slide, x, y, w, value, label, color, sub) {
  slide.addText(value, {
    x, y, w, h: 0.95,
    fontFace: NUM, fontSize: 52, bold: true, color,
    margin: 0, valign: "middle",
  });
  slide.addText(label, {
    x, y: y + 0.95, w, h: 0.3,
    fontFace: CJK, fontSize: 13, bold: true, color: SLATE, margin: 0,
  });
  if (sub) {
    slide.addText(sub, {
      x, y: y + 1.24, w, h: 0.5,
      fontFace: CJK, fontSize: 11, color: MUTED, margin: 0, lineSpacing: 15,
    });
  }
}

function bg(slide, color) {
  slide.background = { color };
}

/** Body text block with real bullets (never a literal •). */
function bullets(slide, items, x, y, w, h, size, color) {
  slide.addText(
    items.map((t, i) => ({
      text: t,
      options: {
        bullet: true,
        breakLine: i < items.length - 1,
        paraSpaceAfter: 8,
      },
    })),
    {
      x, y, w, h,
      fontFace: CJK, fontSize: size || 14, color: color || SLATE,
      margin: 0, valign: "top", lineSpacing: size ? size * 1.5 : 21,
    }
  );
}

// ================================================================ 1 title
{
  const s = pres.addSlide();
  bg(s, INK);

  s.addShape(pres.ShapeType.ellipse, {
    x: M, y: 1.55, w: 0.2, h: 0.2, fill: { color: AMBER },
  });
  s.addText("技术汇报  ·  2026-09", {
    x: M + 0.36, y: 1.52, w: 6, h: 0.3,
    fontFace: CJK, fontSize: 13, color: AMBER, charSpacing: 2, margin: 0,
  });

  s.addText("中文口语评测基准", {
    x: M, y: 2.15, w: W - 2 * M, h: 1.0,
    fontFace: CJK, fontSize: 52, bold: true, color: WHITE, margin: 0,
  });
  s.addText("传统声学 CAPT 引擎  vs  Qwen-Omni 直接评分", {
    x: M, y: 3.25, w: W - 2 * M, h: 0.6,
    fontFace: CJK, fontSize: 22, color: "B9C0C8", margin: 0,
  });

  // Three meta figures on a hairline row (spacing, not stripes).
  const meta = [
    ["124", "条构造性真值语料"],
    ["13×", "提示词带来的误收率倍数"],
    ["0.01s", "传统引擎中位延迟"],
  ];
  meta.forEach(([v, l], i) => {
    const x = M + i * 4.1;
    s.addText(v, {
      x, y: 4.65, w: 3.6, h: 0.6,
      fontFace: NUM, fontSize: 30, bold: true, color: AMBER, margin: 0,
    });
    s.addText(l, {
      x, y: 5.28, w: 3.6, h: 0.3,
      fontFace: CJK, fontSize: 13, color: "8A929B", margin: 0,
    });
  });

  s.addText("github.com/mutsunn1/benchmark-audiotest", {
    x: M, y: 6.55, w: 8, h: 0.3,
    fontFace: "Courier New", fontSize: 11, color: "9AA3AC", margin: 0,
  });

  s.addNotes(
    "开场：这不是一次文献综述，是把调研报告里的两条技术路线真的搭起来跑了一遍。\n" +
    "一句话结论先放这儿：报告说的'语言先验脑补纠正'是可复现的，Omni 放过错误的比例是传统引擎的 2.4 到 3.2 倍。"
  );
}

// ================================================================ 2 问题
{
  const s = pres.addSlide();
  bg(s, PAPER);
  marker(s, "背景", M, 0.5);
  title(s, "要检验的断言", 0.82);

  s.addText("调研报告的论断", {
    x: M, y: 2.15, w: 5.4, h: 0.35,
    fontFace: CJK, fontSize: 16, bold: true, color: SLATE, margin: 0,
  });
  bullets(s, [
    "专业引擎用强制对齐 + GOP 算对数后验概率比，剥离高层语义，只测物理量",
    "Omni 模型带强语言先验，会把不标准发音「脑补」成合理的词，漏检率升高",
    "业界走向级联：声学引擎做 CT，Omni 做主治医生",
  ], M, 2.68, 5.5, 3.5, 15.5);

  // The falsifiable prediction, on an amber-tinted card.
  s.addShape(pres.ShapeType.roundRect, {
    x: 6.9, y: 2.15, w: 5.65, h: 3.9, rectRadius: 0.08,
    fill: { color: "FDF3E2" }, line: { color: "EFD9AE", width: 1 },
  });
  s.addText("可检验的预测", {
    x: 7.25, y: 2.45, w: 5, h: 0.32,
    fontFace: CJK, fontSize: 15, bold: true, color: "9A6B1F", margin: 0,
  });
  s.addText("若「语言先验脑补纠正」成立，则 Omni 的误收率应显著高于传统引擎。", {
    x: 7.25, y: 3.0, w: 4.95, h: 1.25,
    fontFace: CJK, fontSize: 19, color: "6B4A12", margin: 0, lineSpacing: 30,
  });
  s.addText("误收率 = 把错误发音判成正确的比例", {
    x: 7.25, y: 4.5, w: 4.95, h: 0.5,
    fontFace: CJK, fontSize: 12, color: "9A6B1F", margin: 0, lineSpacing: 17,
  });

  s.addText(
    "注意：本基准比的是「能否区分范畴替换」，不是「能否给出口音分数」。见末页局限。",
    { x: M, y: 6.5, w: 11.8, h: 0.4, fontFace: CJK, fontSize: 11.5, color: MUTED, margin: 0 }
  );

  s.addNotes("把报告的定性论断转成一个能测的量：误收率。后面每一页都在回答这一个问题。");
}

// ================================================================ 3 方法
{
  const s = pres.addSlide();
  bg(s, PAPER);
  marker(s, "方法", M, 0.5);
  title(s, "怎么造出带真值的语料", 0.82);

  s.addText("最小对立对构造替换：同一音色朗读对立对的另一侧，真值由构造保证，不需要人工标注。", {
    x: M, y: 1.62, w: 11.8, h: 0.4,
    fontFace: CJK, fontSize: 14, color: MUTED, margin: 0,
  });

  // Substitution diagram: expected -> spoken
  const boxY = 2.35;
  const boxes = [
    { x: M, label: "应该说的", char: "妈", py: "mā", note: "T1 阴平", tone: MOSS },
    { x: M + 2.35, label: "实际说的", char: "麻", py: "má", note: "T2 阳平", tone: BRICK },
  ];
  boxes.forEach((b, i) => {
    s.addShape(pres.ShapeType.roundRect, {
      x: b.x, y: boxY, w: 1.95, h: 1.95, rectRadius: 0.06,
      fill: { color: WHITE }, line: { color: LINE, width: 1 },
    });
    s.addText(b.label, {
      x: b.x, y: boxY + 0.18, w: 1.95, h: 0.28,
      fontFace: CJK, fontSize: 11.5, color: MUTED, align: "center", margin: 0,
    });
    s.addText(b.char, {
      x: b.x, y: boxY + 0.5, w: 1.95, h: 0.85,
      fontFace: CJK, fontSize: 40, bold: true, color: SLATE, align: "center", margin: 0,
    });
    s.addText(b.py + "  " + b.note, {
      x: b.x, y: boxY + 1.42, w: 1.95, h: 0.32,
      fontFace: NUM, fontSize: 13, color: b.tone, align: "center", margin: 0,
    });
  });
  s.addText("→", {
    x: M + 1.95, y: boxY + 0.6, w: 0.4, h: 0.7,
    fontFace: CJK, fontSize: 26, color: MUTED, align: "center", margin: 0,
  });
  s.addText("判为错误 · 维度＝声调 · T1→T2", {
    x: M + 4.5, y: boxY + 0.75, w: 4.2, h: 0.45,
    fontFace: CJK, fontSize: 14, bold: true, color: BRICK, margin: 0,
  });

  // Coverage stats
  stat(s, 8.65, boxY + 0.05, 1.9, "124", "条计分条目", SLATE, "40 正确 / 84 错误");
  stat(s, 10.75, boxY + 0.05, 1.8, "4", "类对立组", SLATE, "声调·声母·韵母·整句");

  // The trap that makes the whole thing valid
  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 4.75, w: 11.85, h: 1.55, rectRadius: 0.08,
    fill: { color: "EDF3EE" }, line: { color: "C8DCCF", width: 1 },
  });
  s.addText("关键设计：learner / teacher 双轨录音", {
    x: M + 0.35, y: 4.98, w: 11.2, h: 0.3,
    fontFace: CJK, fontSize: 15, bold: true, color: "2C5F45", margin: 0,
  });
  s.addText(
    "每条文本合成两条录音：learner 是被评测的，teacher 是作对照的标准发音。" +
    "两条必须不是同一份录音——否则所有 DTW 比对恒为 0 距离、全部判对，基准会自我恭维。" +
    "（实测中踩到过：macOS say 会把语速量化到粗档位，160 和 180 渲染出字节相同的音频。）",
    {
      x: M + 0.35, y: 5.33, w: 11.2, h: 0.9,
      fontFace: CJK, fontSize: 12.5, color: "35594A", margin: 0, lineSpacing: 19,
    }
  );

  s.addNotes(
    "三个要点：\n" +
    "1. 真值靠构造，不靠标注，所以能算出准确率。\n" +
    "2. learner/teacher 必须分开录音，这是整个基准有效性的前提。\n" +
    "3. 声母组用物理线索（谱重心判平翘舌、VOT 判送气），韵母组用 F3 判圆唇、F3-F2 收敛判前后鼻音。"
  );
}

// ================================================================ 4 两个系统
{
  const s = pres.addSlide();
  bg(s, PAPER);
  marker(s, "系统", M, 0.5);
  title(s, "两条路线，同一批音频", 0.82);

  // Left: traditional pipeline
  s.addText("传统 CAPT 引擎（自研，按报告第 2–3 节）", {
    x: M, y: 1.72, w: 5.5, h: 0.32,
    fontFace: CJK, fontSize: 15, bold: true, color: SLATE, margin: 0,
  });
  const steps = [
    "预加重 0.97 → 25ms 汉明窗 / 10ms 帧移",
    "26 梅尔滤波器 → 39 维 MFCC",
    "Praat 自相关基频跟踪 F0",
    "模板 DTW 强制对齐 → 音节 [ts, te]",
    "石锋五度 T 值归一化 → 声调原型 DTW",
    "段音物理线索：谱重心 / VOT / F3",
  ];
  steps.forEach((t, i) => {
    const y = 2.2 + i * 0.52;
    s.addText(String(i + 1), {
      x: M, y, w: 0.3, h: 0.3,
      fontFace: NUM, fontSize: 13, bold: true, color: AMBER, margin: 0,
    });
    s.addText(t, {
      x: M + 0.38, y, w: 5.2, h: 0.3,
      fontFace: CJK, fontSize: 13, color: SLATE, margin: 0,
    });
  });
  s.addText("测的是物理量：对数后验概率比、T 值轨迹、嗓音起始时间", {
    x: M, y: 5.6, w: 5.5, h: 0.5,
    fontFace: CJK, fontSize: 12, italic: true, color: MUTED, margin: 0, lineSpacing: 17,
  });

  // Right: omni
  s.addText("Qwen-Omni（qwen3.5-omni-flash）", {
    x: 7.1, y: 1.72, w: 5.5, h: 0.32,
    fontFace: CJK, fontSize: 15, bold: true, color: SLATE, margin: 0,
  });
  const modes = [
    ["open", "「转写你听到的音节」", "纯感知能力"],
    ["context", "只给交际场景，不给目标文本", "语义是否拉偏转写"],
    ["scripted", "给出目标文本，问发音对不对", "标准 CAPT 场景，陷阱在此"],
  ];
  modes.forEach(([name, prompt, what], i) => {
    const y = 2.2 + i * 1.06;
    s.addShape(pres.ShapeType.roundRect, {
      x: 7.1, y, w: 5.45, h: 0.9, rectRadius: 0.06,
      fill: { color: WHITE }, line: { color: LINE, width: 1 },
    });
    s.addText(name, {
      x: 7.32, y: y + 0.12, w: 1.3, h: 0.28,
      fontFace: "Courier New", fontSize: 13, bold: true, color: AMBER, margin: 0,
    });
    s.addText(prompt, {
      x: 8.6, y: y + 0.12, w: 3.8, h: 0.28,
      fontFace: CJK, fontSize: 12.5, color: SLATE, margin: 0,
    });
    s.addText(what, {
      x: 7.32, y: y + 0.46, w: 5, h: 0.3,
      fontFace: CJK, fontSize: 11.5, color: MUTED, margin: 0,
    });
  });
  s.addText("三档跑同一段音频：才能区分「没听清」和「听清了但顺从提示词」", {
    x: 7.1, y: 5.6, w: 5.5, h: 0.5,
    fontFace: CJK, fontSize: 12, italic: true, color: MUTED, margin: 0, lineSpacing: 17,
  });

  s.addNotes(
    "两边面对的是同一批音频、同一个评测集，唯一变量是评分方法。\n" +
    "三档提示词是本次实验设计的核心：只跑 scripted 的话，'模型没听清'和'模型顺从了提示词'长得一模一样。"
  );
}

// ================================================================ 5 主结果
{
  const s = pres.addSlide();
  bg(s, PAPER);
  marker(s, "结果", M, 0.5);
  title(s, "同一个模型，只差提示词里的一行字", 0.82);

  s.addText(
    "同说话人条件，124 条语料。两次运行用的是同一个模型、同一批音频，唯一差别是 scripted 档的提示词里写了目标文本。",
    { x: M, y: 1.6, w: 11.8, h: 0.4, fontFace: CJK, fontSize: 13, color: MUTED, margin: 0 }
  );

  s.addChart(
    pres.ChartType.bar,
    [
      {
        name: "误收率 FAR",
        labels: ["传统 CAPT", "Omni open", "Omni scripted"],
        values: [6.0, 3.8, 50.0],
      },
    ],
    {
      x: M, y: 2.05, w: 6.9, h: 3.9,
      barDir: "bar",
      showTitle: false,
      showValue: true,
      dataLabelPosition: "outEnd",
      dataLabelColor: SLATE,
      dataLabelFontFace: NUM,
      dataLabelFontSize: 15,
      dataLabelFormatCode: '0.0"%"',
      chartColors: [MOSS, MOSS, BRICK],
      varyColors: true,
      valAxisMaxVal: 60,
      valAxisLabelColor: MUTED,
      valAxisLabelFontFace: NUM,
      valAxisLabelFontSize: 11,
      valGridLine: { color: LINE, size: 1 },
      catAxisLabelColor: SLATE,
      catAxisLabelFontFace: CJK,
      catAxisLabelFontSize: 13,
      catGridLine: { style: "none" },
      showLegend: false,
      barGapWidthPct: 55,
    }
  );

  stat(s, 8.35, 2.15, 4.2, "3.8%", "Omni open 误收率", MOSS, "不给目标文本时");
  stat(s, 8.35, 4.25, 4.2, "50.0%", "Omni scripted 误收率", BRICK, "给出目标文本时");

  s.addText([
    { text: "声调识别率同步从 75.0% 掉到 23.3%", options: { breakLine: true } },
    { text: "——提示词里的目标文本污染了它的感知。", options: {} },
  ], {
    x: 8.35, y: 6.05, w: 4.3, h: 0.75,
    fontFace: CJK, fontSize: 12.5, color: SLATE, margin: 0, lineSpacing: 18,
  });

  s.addNotes(
    "全场最关键的一页。误收率 = 读错的音频被判成正确的比例。\n" +
    "同一个模型、同一批音频，只是提示词里多写了一行目标文本：误收率 3.8% -> 50.0%，13 倍。\n" +
    "而且 open 档的误收率（3.8%）比传统引擎（6.0%）还低——说明模型本身听得很好，是提示词把它带偏了。"
  );
}

// ================================================================ 6 完整表
{
  const s = pres.addSlide();
  bg(s, PAPER);
  marker(s, "结果", M, 0.5);
  title(s, "同说话人条件完整指标（124 条）", 0.82);

  const head = ["系统", "判定准确率", "误收率 FAR", "误拒率 FRR", "声调识别率", "错误定位率", "中位延迟"];
  const rows = [
    ["Omni open（不给文本）", "93.9%", "3.8%", "11.1%", "75.0%", "89.5%", "0.68s"],
    ["传统 CAPT", "88.7%", "6.0%", "22.5%", "69.0%", "93.7%", "0.01s"],
    ["Omni scripted（给文本）", "56.1%", "50.0%", "30.8%", "23.3%", "66.7%", "1.19s"],
  ];

  const tableRows = [
    head.map((h) => ({
      text: h,
      options: {
        bold: true, color: WHITE, fill: { color: SLATE },
        fontFace: CJK, fontSize: 12.5, align: "center", valign: "middle",
      },
    })),
    ...rows.map((r, ri) =>
      r.map((c, ci) => ({
        text: c,
        options: {
          color: ci === 2 ? (ri === 2 ? BRICK : MOSS) : SLATE,
          bold: ci === 2,
          fill: { color: ri % 2 ? WHITE : "F1F2F0" },
          fontFace: ci === 0 ? CJK : NUM,
          fontSize: ci === 0 ? 12.5 : 13,
          align: ci === 0 ? "left" : "center",
          valign: "middle",
        },
      }))
    ),
  ];

  s.addTable(tableRows, {
    x: M, y: 1.8, w: 11.85,
    colW: [3.35, 1.45, 1.45, 1.35, 1.45, 1.45, 1.35],
    rowH: 0.62,
    border: { type: "solid", color: LINE, pt: 1 },
    autoPage: false,
    margin: 0.08,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 4.55, w: 11.85, h: 2.0, rectRadius: 0.08,
    fill: { color: "FDF3E2" }, line: { color: "EFD9AE", width: 1 },
  });
  s.addText("怎么读这张表", {
    x: M + 0.35, y: 4.76, w: 11, h: 0.3,
    fontFace: CJK, fontSize: 14, bold: true, color: "9A6B1F", margin: 0,
  });
  s.addText([
    {
      text: "纯转写的 Omni 是这一轮最好的系统——准确率、误收率、声调识别三项都优于传统引擎。",
      options: { breakLine: true },
    },
    {
      text: "但一旦把目标文本写进提示词，它从最好掉到最差。同一个模型的两种用法，差距比两个模型的差距还大。",
      options: { breakLine: true },
    },
    {
      text: "传统引擎的强项是错误定位率（93.7%）——它不只知道错了，还知道错在声母、韵母还是声调。",
      options: {},
    },
  ], {
    x: M + 0.35, y: 5.12, w: 11.15, h: 1.3,
    fontFace: CJK, fontSize: 12.5, color: "6B4A12", margin: 0, lineSpacing: 19,
  });

  s.addNotes("准确率、误收率、声调识别三项 Omni open 都赢；错误定位传统引擎赢。分工很清楚。");
}

// ================================================================ 7 陷阱
{
  const s = pres.addSlide();
  bg(s, INK);
  marker(s, "实验", M, 0.5, AMBER);
  title(s, "语言陷阱：0% → 100%", 0.82, WHITE);

  s.addText("8 条整句：语境强烈预测某个词（我想吃水饺），音频实际说的是其最小对立伙伴（我想吃睡觉）。", {
    x: M, y: 1.62, w: 11.8, h: 0.35,
    fontFace: CJK, fontSize: 13, color: "9AA3AC", margin: 0,
  });

  const traps = [
    ["Omni open", "0.0%", "不给目标文本", MOSS],
    ["Omni context", "25.0%", "只给交际场景", AMBER],
    ["Omni scripted", "100.0%", "给出目标文本", BRICK],
    ["传统 CAPT", "0.0%", "自研声学引擎", MOSS],
  ];
  s.addText("陷阱句误收率", {
    x: M, y: 2.2, w: 5.5, h: 0.3,
    fontFace: CJK, fontSize: 13, bold: true, color: AMBER, margin: 0,
  });
  traps.forEach(([name, far, note, color], i) => {
    const y = 2.62 + i * 0.66;
    s.addText(name, {
      x: M, y, w: 2.4, h: 0.42,
      fontFace: CJK, fontSize: 14, color: WHITE, margin: 0, valign: "middle",
    });
    s.addText(far, {
      x: M + 2.5, y, w: 1.7, h: 0.42,
      fontFace: NUM, fontSize: 22, bold: true, color, margin: 0, valign: "middle",
    });
    s.addText(note, {
      x: M + 4.3, y, w: 3.0, h: 0.42,
      fontFace: CJK, fontSize: 12, color: "8A929B", margin: 0, valign: "middle",
    });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 7.6, y: 2.2, w: 5.05, h: 3.15, rectRadius: 0.08,
    fill: { color: "23262C" }, line: { color: "3A3F47", width: 1 },
  });
  s.addText("这条曲线说明什么", {
    x: 7.9, y: 2.42, w: 4.5, h: 0.3,
    fontFace: CJK, fontSize: 14, bold: true, color: AMBER, margin: 0,
  });
  s.addText([
    { text: "不给文本：全对。模型听到了替换，如实转写。", options: { breakLine: true } },
    { text: "只给场景：放过 1/4。语义开始起作用。", options: { breakLine: true } },
    { text: "给出文本：8 条全放过。提示词完全压倒了音频。", options: {} },
  ], {
    x: 7.9, y: 2.85, w: 4.5, h: 2.3,
    fontFace: CJK, fontSize: 12.5, color: "C9D1D9", margin: 0, lineSpacing: 20,
  });

  s.addText("陷阱的强度随提示词给出的信息量单调上升。", {
    x: M, y: 5.75, w: 11.8, h: 0.4,
    fontFace: CJK, fontSize: 14, bold: true, color: AMBER, margin: 0,
  });

  s.addNotes(
    "最干净的一组证据。陷阱句误收率随提示词信息量单调上升：0% -> 25% -> 100%。\n" +
    "注意传统引擎也是 0%，但它的对照句误拒率是 100%——它是靠'什么都判错'换来的低误收，不是真本事。"
  );
}

// ================================================================ 8 一致性
{
  const s = pres.addSlide();
  bg(s, PAPER);
  marker(s, "工程", M, 0.5);
  title(s, "产品化的两条硬约束", 0.82);

  stat(s, M, 1.9, 5.2, "70%", "声调一致率", BRICK,
       "同一段音频、同一提示词，重复评测 3 次。判定一致率 80%。");
  stat(s, 7.1, 1.9, 5.2, "~100×", "延迟差距", BRICK,
       "Omni 1.19s  vs  传统引擎 0.01s（中位，单条）。");

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 4.35, w: 11.85, h: 2.05, rectRadius: 0.08,
    fill: { color: WHITE }, line: { color: LINE, width: 1 },
  });
  s.addText("对「跟读打分」意味着什么", {
    x: M + 0.4, y: 4.58, w: 11, h: 0.32,
    fontFace: CJK, fontSize: 15, bold: true, color: SLATE, margin: 0,
  });
  bullets(s, [
    "同一段录音两次评测给出不同分数，学习者会认为系统在随机打分——这是信任问题，不是精度问题",
    "传统引擎是确定性数学映射，重复评测结果必然一致；Omni 是自回归采样，稳定性需要单独实测",
    "秒级延迟 + 昂贵的音频 token，在高并发的跟读场景下成本效益比差",
  ], M + 0.4, 4.98, 11.1, 1.35, 12.5);

  s.addNotes("报告里说 Omni「有幻觉风险和延迟成本」，这次量化了：声调 30% 的概率给出不同答案，延迟差 100 倍。");
}

// ================================================================ 9 跨说话人
{
  const s = pres.addSlide();
  bg(s, PAPER);
  marker(s, "边界条件", M, 0.5);
  title(s, "换个说话人，传统引擎掉了 24 个点", 0.82);

  s.addText(
    "把 TTS 换成 Qwen（learner=Cherry，teacher=Ethan，真正的跨说话人）后重跑同一套语料。",
    { x: M, y: 1.6, w: 11.8, h: 0.4, fontFace: CJK, fontSize: 13, color: MUTED, margin: 0 }
  );

  // Degradation bars
  const bars = [
    ["传统 CAPT", "88.7%", "64.5%", "−24.2", BRICK],
    ["Omni open", "93.9%", "84.0%", "−9.9", AMBER],
    ["Omni scripted", "56.1%", "58.9%", "+2.8", MOSS],
  ];
  s.addText("同说话人 → 跨说话人 判定准确率", {
    x: M, y: 2.2, w: 11.8, h: 0.3,
    fontFace: CJK, fontSize: 13, bold: true, color: SLATE, margin: 0,
  });
  bars.forEach(([name, before, after, delta, color], i) => {
    const y = 2.68 + i * 0.72;
    s.addText(name, {
      x: M, y, w: 2.4, h: 0.5,
      fontFace: CJK, fontSize: 13.5, color: SLATE, margin: 0, valign: "middle",
    });
    s.addText(before, {
      x: M + 2.5, y, w: 1.3, h: 0.5,
      fontFace: NUM, fontSize: 17, color: MUTED, margin: 0, valign: "middle",
    });
    s.addText("→", {
      x: M + 3.75, y, w: 0.4, h: 0.5,
      fontFace: CJK, fontSize: 14, color: MUTED, align: "center", margin: 0, valign: "middle",
    });
    s.addText(after, {
      x: M + 4.15, y, w: 1.4, h: 0.5,
      fontFace: NUM, fontSize: 20, bold: true, color: SLATE, margin: 0, valign: "middle",
    });
    s.addText(delta, {
      x: M + 5.55, y, w: 1.5, h: 0.5,
      fontFace: NUM, fontSize: 17, bold: true, color, margin: 0, valign: "middle",
    });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 7.3, y: 2.62, w: 5.35, h: 3.5, rectRadius: 0.08,
    fill: { color: "FDF3E2" }, line: { color: "EFD9AE", width: 1 },
  });
  s.addText("为什么掉这么多", {
    x: 7.6, y: 2.84, w: 4.8, h: 0.3,
    fontFace: CJK, fontSize: 14, bold: true, color: "9A6B1F", margin: 0,
  });
  s.addText([
    {
      text: "传统引擎的声调原型是从 teacher 的正确读音拟合的。实测 teacher（Ethan）的孤立字：",
      options: { breakLine: true },
    },
    {
      text: "T1 原型 3.5→2.9、T2 原型 1.9→2.0（几乎不升），且 T1 类内离散度 0.97。",
      options: { breakLine: true },
    },
    {
      text: "原型本身是糊的，判定就散。参考发音不够标准时，「做 CT」的那一环就不准了。",
      options: {},
    },
  ], {
    x: 7.6, y: 3.24, w: 4.8, h: 2.6,
    fontFace: CJK, fontSize: 12.5, color: "6B4A12", margin: 0, lineSpacing: 20,
  });

  s.addText("报告推荐的级联架构有一个没写出来的前提：参考发音必须足够标准。", {
    x: M, y: 6.35, w: 11.8, h: 0.4,
    fontFace: CJK, fontSize: 13.5, bold: true, color: SLATE, margin: 0,
  });

  s.addNotes(
    "这是我觉得最有价值的意外发现。传统引擎跨说话人掉了 24 个点，误拒率从 22.5% 涨到 75%。\n" +
    "查了原因不是 bug：Ethan 这个音色的孤立字声调实现本身区分度就低。\n" +
    "结论：'声学引擎做 CT' 的前提是参考发音本身够标准。这个前提报告里没提。"
  );
}

// ================================================================ 10 结论
{
  const s = pres.addSlide();
  bg(s, INK);
  marker(s, "结论", M, 0.5, AMBER);
  title(s, "结论", 0.82, WHITE);

  const conclusions = [
    ["01", "语言陷阱被隔离出来了", "同一模型、同一批音频，只因为提示词写了目标文本：误收率 3.8% → 50.0%，声调识别 75.0% → 23.3%。陷阱句上 0% → 100%。"],
    ["02", "问题不在模型，在提示词", "纯转写的 Omni 准确率 93.9%，比传统引擎（88.7%）还高。模型听得很好——是提示词把它带偏了。"],
    ["03", "报告推荐的级联有前提", "参考发音本身不标准时，传统引擎跨说话人从 88.7% 掉到 64.5%。「做 CT」的那一环会先坏掉。"],
    ["04", "选型看场景，不是看分数", "高并发跟读打分：传统引擎（确定、毫秒级、定位准）。自由对话与错因解释：Omni。两者不是同一张考卷。"],
  ];
  conclusions.forEach(([n, head, body], i) => {
    const y = 1.85 + i * 1.22;
    s.addText(n, {
      x: M, y, w: 0.7, h: 0.5,
      fontFace: NUM, fontSize: 24, bold: true, color: AMBER, margin: 0,
    });
    s.addText(head, {
      x: M + 0.8, y: y - 0.02, w: 10.8, h: 0.36,
      fontFace: CJK, fontSize: 17, bold: true, color: WHITE, margin: 0,
    });
    s.addText(body, {
      x: M + 0.8, y: y + 0.36, w: 10.8, h: 0.62,
      fontFace: CJK, fontSize: 12.5, color: "9AA3AC", margin: 0, lineSpacing: 18,
    });
  });

  s.addNotes("四条结论。第 2 条是这次最反直觉的：问题不在模型能力，在提示词设计。");
}

// ================================================================ 11 局限与复现
{
  const s = pres.addSlide();
  bg(s, PAPER);
  marker(s, "边界", M, 0.5);
  title(s, "局限与复现", 0.82);

  s.addText("这套数字不能说明什么", {
    x: M, y: 1.75, w: 6.6, h: 0.32,
    fontFace: CJK, fontSize: 15, bold: true, color: BRICK, margin: 0,
  });
  bullets(s, [
    "语料是 TTS 合成的干净范畴替换，不是真实学习者的连续渐变偏误",
    "传统引擎是按报告自研的参考实现，绝对水平不代表讯飞 / SpeechSuper",
    "跨说话人条件只测了一对音色（Cherry → Ethan），换音色结论可能变",
    "context 档 n=8，样本量小，只作方向性参考",
  ], M, 2.2, 6.4, 3.4, 12.5);

  s.addText("复现", {
    x: 7.7, y: 1.75, w: 5.0, h: 0.32,
    fontFace: CJK, fontSize: 15, bold: true, color: MOSS, margin: 0,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 7.7, y: 2.2, w: 4.9, h: 2.25, rectRadius: 0.06,
    fill: { color: "1E2126" }, line: { color: "33383F", width: 1 },
  });
  s.addText(
    [
      { text: "uv sync", options: { breakLine: true } },
      { text: "uv run captbench run \\", options: { breakLine: true } },
      { text: "  --tts qwen \\", options: { breakLine: true } },
      { text: "  --scorers traditional,omni", options: { breakLine: true } },
      { text: "", options: { breakLine: true } },
      { text: "uv run pytest -q   # 101 passed", options: {} },
    ],
    {
      x: 7.95, y: 2.4, w: 4.5, h: 1.9,
      fontFace: "Courier New", fontSize: 11.5, color: "C9D1D9",
      margin: 0, lineSpacing: 19,
    }
  );

  s.addText("github.com/mutsunn1/benchmark-audiotest", {
    x: 7.7, y: 4.6, w: 5.0, h: 0.3,
    fontFace: "Courier New", fontSize: 11, color: MUTED, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 5.0, w: 11.9, h: 1.5, rectRadius: 0.08,
    fill: { color: "EDF3EE" }, line: { color: "C8DCCF", width: 1 },
  });
  s.addText("一句话", {
    x: M + 0.35, y: 5.2, w: 11, h: 0.3,
    fontFace: CJK, fontSize: 13, bold: true, color: "2C5F45", margin: 0,
  });
  s.addText(
    "调研报告关于「语言先验导致漏检」的判断得到了量化验证，而且比报告的估计更严重——" +
    "在陷阱句上它是 100%，不是「漏检率上升」。但补救办法可能比报告设想的简单：问题出在提示词，不在模型。",
    {
      x: M + 0.35, y: 5.55, w: 11.2, h: 0.85,
      fontFace: CJK, fontSize: 13, color: "35594A", margin: 0, lineSpacing: 20,
    }
  );

  s.addNotes("把边界讲清楚，结论才可信。特别强调：这不是真实学习者语料，传统引擎也不是商业引擎。");
}

pres.writeFile({ fileName: "/tmp/captbench-deck/deck.pptx" }).then((f) => {
  console.log("wrote " + f);
});
