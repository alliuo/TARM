
module mul8_123 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[7] , \intadd_0/SUM[3] , \intadd_0/SUM[2] , n204, n205,
         n206, n207, n208, n209, n210, n211, n212, n213, n214, n215, n216,
         n217, n218, n219, n220, n221, n222, n223, n224, n225, n226, n227,
         n228, n229, n230, n231, n232, n233, n234, n235, n236, n237, n238,
         n239, n240, n241, n242, n243, n244, n245, n246, n247, n248, n249,
         n250, n251, n252, n253, n254, n255, n256, n257, n258, n259, n260,
         n261, n262, n263, n264, n265, n266, n267, n268, n269, n270, n271,
         n272, n273, n274, n275, n276, n277, n278, n279, n280, n281, n282,
         n283, n284, n285, n286, n287, n288, n289, n290, n291, n292, n293,
         n294, n295, n296, n297, n298, n299, n300, n301, n302, n303, n304,
         n305, n306, n307, n308, n309, n310, n311, n312, n313, n314, n315,
         n316, n317, n318, n319, n320, n321, n322, n323, n324, n325, n326,
         n327, n328, n329, n330, n331, n332, n333, n334, n335, n336, n337,
         n338, n339, n340, n341, n342, n343, n344, n345, n346, n347, n348,
         n349, n350, n351, n352, n353, n354, n355, n356, n357, n358, n359,
         n360, n361, n362, n363, n364, n365, n366, n367, n368, n369, n370,
         n371, n372, n373, n374, n375, n376, n377, n378, n379, n380, n381,
         n382, n383, n384, n385, n386, n387, n388, n389, n390;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U195 ( .A1(n342), .B1(n340), .ZN(n343) );
  VHSR_NOR2_1 U196 ( .A1(n335), .A2(n334), .ZN(n333) );
  VHSR_INOR3_2 U197 ( .A1(n231), .B1(n312), .B2(n273), .ZN(n291) );
  VHSR_NOR2_1 U198 ( .A1(n386), .A2(n385), .ZN(n384) );
  VHSR_INOR2_2 U199 ( .A1(n351), .B1(n350), .ZN(n382) );
  VHSR_INOR2_2 U200 ( .A1(n378), .B1(n377), .ZN(product[2]) );
  VHSR_CLKN_1 U201 ( .I(n347), .ZN(product[13]) );
  VHSR_NOR2_2 U202 ( .A1(n309), .A2(n310), .ZN(n320) );
  VHSR_AD1_1 U203 ( .A(n359), .B(n358), .CI(n357), .CO(n354), .S(product[9])
         );
  VHSR_AD1_1 U204 ( .A(n384), .B(n367), .CI(n366), .CO(n368), .S(product[5])
         );
  VHSR_AD1_1 U205 ( .A(n365), .B(n364), .CI(n363), .CO(n360), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U206 ( .A(n362), .B(n361), .CI(n360), .CO(n357), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U207 ( .A(n356), .B(n355), .CI(n354), .CO(n371), .S(product[10])
         );
  VHSR_IN_2 U208 ( .I(b[2]), .ZN(n309) );
  VHSR_IN_2 U209 ( .I(a[2]), .ZN(n310) );
  VHSR_IN_2 U210 ( .I(b[0]), .ZN(n388) );
  VHSR_IN_2 U211 ( .I(a[0]), .ZN(n389) );
  VHSR_NOR2_1 U212 ( .A1(n388), .A2(n389), .ZN(product[0]) );
  VHSR_IN_2 U213 ( .I(b[1]), .ZN(n390) );
  VHSR_NOR2_1 U214 ( .A1(n390), .A2(n389), .ZN(n204) );
  VHSR_IN_2 U215 ( .I(a[1]), .ZN(n387) );
  VHSR_NOR2_1 U216 ( .A1(n309), .A2(n387), .ZN(n205) );
  VHSR_AOI22_2 U217 ( .A1(n320), .A2(product[0]), .B1(n204), .B2(n205), .ZN(
        n378) );
  VHSR_AOI22_2 U218 ( .A1(b[1]), .A2(a[2]), .B1(b[0]), .B2(a[3]), .ZN(n308) );
  VHSR_AOI21_2 U219 ( .A1(a[0]), .A2(b[3]), .B(n205), .ZN(n307) );
  VHSR_IN_2 U220 ( .I(n206), .ZN(product[3]) );
  VHSR_AOI22_2 U221 ( .A1(a[6]), .A2(b[3]), .B1(a[7]), .B2(b[2]), .ZN(n242) );
  VHSR_IN_2 U222 ( .I(b[3]), .ZN(n312) );
  VHSR_IN_2 U223 ( .I(a[5]), .ZN(n278) );
  VHSR_IN_2 U224 ( .I(a[4]), .ZN(n277) );
  VHSR_NOR4_2 U225 ( .A1(n312), .A2(n309), .A3(n278), .A4(n277), .ZN(n240) );
  VHSR_IN_2 U226 ( .I(a[7]), .ZN(n273) );
  VHSR_NOR2_1 U227 ( .A1(n273), .A2(n390), .ZN(n208) );
  VHSR_AOI211_2 U228 ( .A1(b[2]), .A2(a[4]), .B(n312), .C(n278), .ZN(n209) );
  VHSR_CLKNAND2_2 U229 ( .A1(a[6]), .A2(b[2]), .ZN(n211) );
  VHSR_IN_2 U230 ( .I(n211), .ZN(n207) );
  VHSR_MAOI222_2 U231 ( .A(n208), .B(n209), .C(n207), .ZN(n221) );
  VHSR_AOI21_2 U232 ( .A1(b[1]), .A2(a[7]), .B(n209), .ZN(n212) );
  VHSR_IN_2 U233 ( .I(n221), .ZN(n210) );
  VHSR_AOI21_2 U234 ( .A1(n212), .A2(n211), .B(n210), .ZN(n249) );
  VHSR_CLKNAND2_2 U235 ( .A1(a[6]), .A2(b[1]), .ZN(n218) );
  VHSR_IN_2 U236 ( .I(n218), .ZN(n215) );
  VHSR_NOR4_2 U237 ( .A1(n278), .A2(n277), .A3(n390), .A4(n388), .ZN(n267) );
  VHSR_CLKNAND2_2 U238 ( .A1(b[2]), .A2(a[5]), .ZN(n214) );
  VHSR_CLKNAND2_2 U239 ( .A1(b[3]), .A2(a[4]), .ZN(n213) );
  VHSR_AOI21_2 U240 ( .A1(n214), .A2(n213), .B(n240), .ZN(n216) );
  VHSR_MAOI222_2 U241 ( .A(n215), .B(n267), .C(n216), .ZN(n220) );
  VHSR_CLKNAND2_2 U242 ( .A1(b[2]), .A2(a[4]), .ZN(n263) );
  VHSR_OAI21_2 U243 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n262) );
  VHSR_OAI211_2 U244 ( .A1(n277), .A2(n388), .B(a[5]), .C(b[1]), .ZN(n261) );
  VHSR_MAOI222_2 U245 ( .A(n263), .B(n262), .C(n261), .ZN(n260) );
  VHSR_NOR2_1 U246 ( .A1(n267), .A2(n216), .ZN(n219) );
  VHSR_IN_2 U247 ( .I(n220), .ZN(n217) );
  VHSR_AOI21_2 U248 ( .A1(n219), .A2(n218), .B(n217), .ZN(n252) );
  VHSR_CLKNAND2_2 U249 ( .A1(n260), .A2(n252), .ZN(n251) );
  VHSR_CLKNAND2_2 U250 ( .A1(n220), .A2(n251), .ZN(n248) );
  VHSR_CLKNAND2_2 U251 ( .A1(n249), .A2(n248), .ZN(n247) );
  VHSR_CLKNAND2_2 U252 ( .A1(n221), .A2(n247), .ZN(n239) );
  VHSR_NOR2_1 U253 ( .A1(n240), .A2(n239), .ZN(n238) );
  VHSR_NOR2_1 U254 ( .A1(n242), .A2(n238), .ZN(n231) );
  VHSR_IN_2 U255 ( .I(b[7]), .ZN(n275) );
  VHSR_IN_2 U256 ( .I(a[3]), .ZN(n313) );
  VHSR_IN_2 U257 ( .I(b[6]), .ZN(n276) );
  VHSR_OAI22_2 U258 ( .A1(n276), .A2(n313), .B1(n275), .B2(n310), .ZN(n237) );
  VHSR_AOI22_2 U259 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n228) );
  VHSR_CLKNAND2_2 U260 ( .A1(b[4]), .A2(a[2]), .ZN(n259) );
  VHSR_NAND3_2 U261 ( .A1(a[3]), .A2(b[5]), .A3(n259), .ZN(n227) );
  VHSR_CLKNAND2_2 U262 ( .A1(b[7]), .A2(a[2]), .ZN(n222) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[6]), .A2(a[1]), .ZN(n224) );
  VHSR_OAI22_2 U264 ( .A1(n228), .A2(n227), .B1(n222), .B2(n224), .ZN(n229) );
  VHSR_IN_2 U265 ( .I(b[4]), .ZN(n327) );
  VHSR_OAI211_2 U266 ( .A1(n327), .A2(n389), .B(b[5]), .C(a[1]), .ZN(n258) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[6]), .A2(a[0]), .ZN(n257) );
  VHSR_MAOI222_2 U268 ( .A(n259), .B(n258), .C(n257), .ZN(n256) );
  VHSR_NAND4_2 U269 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n234) );
  VHSR_IN_2 U270 ( .I(b[5]), .ZN(n272) );
  VHSR_OAI22_2 U271 ( .A1(n327), .A2(n313), .B1(n272), .B2(n310), .ZN(n223) );
  VHSR_AND2_2 U272 ( .A1(n234), .A2(n223), .Z(n226) );
  VHSR_OAI21_2 U273 ( .A1(n275), .A2(n389), .B(n224), .ZN(n225) );
  VHSR_NOR4_2 U274 ( .A1(n327), .A2(n272), .A3(n387), .A4(n389), .ZN(n265) );
  VHSR_AND2_2 U275 ( .A1(n256), .A2(n255), .Z(n254) );
  VHSR_AD1_1 U276 ( .A(n226), .B(n225), .CI(n265), .CO(n243), .S(n255) );
  VHSR_AOI21_2 U277 ( .A1(n228), .A2(n227), .B(n229), .ZN(n246) );
  VHSR_OAI32_2 U278 ( .A1(n229), .A2(n254), .A3(n243), .B1(n246), .B2(n229), 
        .ZN(n235) );
  VHSR_CLKNAND2_2 U279 ( .A1(n235), .A2(n234), .ZN(n233) );
  VHSR_CLKNAND2_2 U280 ( .A1(n237), .A2(n233), .ZN(n232) );
  VHSR_NOR3_2 U281 ( .A1(n275), .A2(n313), .A3(n232), .ZN(n290) );
  VHSR_NOR2_1 U282 ( .A1(n312), .A2(n273), .ZN(n230) );
  VHSR_IAO21_2 U283 ( .A1(n231), .A2(n230), .B(n291), .ZN(n294) );
  VHSR_OAI32_2 U284 ( .A1(n290), .A2(n313), .A3(n275), .B1(n232), .B2(n290), 
        .ZN(n293) );
  VHSR_OAI21_2 U285 ( .A1(n235), .A2(n234), .B(n233), .ZN(n236) );
  VHSR_XNOR2_2 U286 ( .A1(n237), .A2(n236), .ZN(n301) );
  VHSR_AOI21_2 U287 ( .A1(n240), .A2(n239), .B(n238), .ZN(n241) );
  VHSR_XNOR2_2 U288 ( .A1(n242), .A2(n241), .ZN(n300) );
  VHSR_NOR2_1 U289 ( .A1(n254), .A2(n243), .ZN(n245) );
  VHSR_AOI22_2 U290 ( .A1(n254), .A2(n243), .B1(n246), .B2(n245), .ZN(n244) );
  VHSR_OAI21_2 U291 ( .A1(n246), .A2(n245), .B(n244), .ZN(n306) );
  VHSR_OAI21_2 U292 ( .A1(n249), .A2(n248), .B(n247), .ZN(n250) );
  VHSR_IN_2 U293 ( .I(n250), .ZN(n305) );
  VHSR_OAI21_2 U294 ( .A1(n260), .A2(n252), .B(n251), .ZN(n253) );
  VHSR_IN_2 U295 ( .I(n253), .ZN(n317) );
  VHSR_IAO21_2 U296 ( .A1(n256), .A2(n255), .B(n254), .ZN(n316) );
  VHSR_AOI31_2 U297 ( .A1(n259), .A2(n258), .A3(n257), .B(n256), .ZN(n323) );
  VHSR_AOI31_2 U298 ( .A1(n263), .A2(n262), .A3(n261), .B(n260), .ZN(n322) );
  VHSR_CLKNAND2_2 U299 ( .A1(b[5]), .A2(a[0]), .ZN(n264) );
  VHSR_OAI32_2 U300 ( .A1(n265), .A2(n387), .A3(n327), .B1(n264), .B2(n265), 
        .ZN(n332) );
  VHSR_CLKNAND2_2 U301 ( .A1(a[4]), .A2(b[4]), .ZN(n280) );
  VHSR_IN_2 U302 ( .I(n280), .ZN(n361) );
  VHSR_CLKNAND2_2 U303 ( .A1(n361), .A2(product[0]), .ZN(n329) );
  VHSR_IN_2 U304 ( .I(n329), .ZN(n331) );
  VHSR_CLKNAND2_2 U305 ( .A1(a[4]), .A2(b[1]), .ZN(n266) );
  VHSR_OAI32_2 U306 ( .A1(n267), .A2(n278), .A3(n388), .B1(n266), .B2(n267), 
        .ZN(n330) );
  VHSR_CLKNAND2_2 U307 ( .A1(a[6]), .A2(b[6]), .ZN(n352) );
  VHSR_IN_2 U308 ( .I(n352), .ZN(n379) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[6]), .A2(b[4]), .ZN(n298) );
  VHSR_NAND3_2 U310 ( .A1(a[7]), .A2(b[5]), .A3(n298), .ZN(n269) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[4]), .A2(b[6]), .ZN(n297) );
  VHSR_NAND3_2 U312 ( .A1(b[7]), .A2(a[5]), .A3(n297), .ZN(n268) );
  VHSR_CLKNAND2_2 U313 ( .A1(n269), .A2(n268), .ZN(n271) );
  VHSR_MAOI222_2 U314 ( .A(n352), .B(n269), .C(n268), .ZN(n336) );
  VHSR_IN_2 U315 ( .I(n336), .ZN(n270) );
  VHSR_OAI21_2 U316 ( .A1(n379), .A2(n271), .B(n270), .ZN(n286) );
  VHSR_NOR3_2 U317 ( .A1(n278), .A2(n272), .A3(n280), .ZN(n302) );
  VHSR_NOR3_2 U318 ( .A1(n273), .A2(n298), .A3(n272), .ZN(n344) );
  VHSR_AOI22_2 U319 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n274) );
  VHSR_NOR2_1 U320 ( .A1(n344), .A2(n274), .ZN(n282) );
  VHSR_NOR4_2 U321 ( .A1(n278), .A2(n277), .A3(n276), .A4(n275), .ZN(n342) );
  VHSR_AOI22_2 U322 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n279) );
  VHSR_NOR2_1 U323 ( .A1(n342), .A2(n279), .ZN(n281) );
  VHSR_NAND3_2 U324 ( .A1(b[5]), .A2(a[5]), .A3(n280), .ZN(n296) );
  VHSR_MAOI222_2 U325 ( .A(n298), .B(n297), .C(n296), .ZN(n295) );
  VHSR_AND2_2 U326 ( .A1(n288), .A2(n295), .Z(n287) );
  VHSR_AD1_1 U327 ( .A(n302), .B(n282), .CI(n281), .CO(n283), .S(n288) );
  VHSR_NOR2_1 U328 ( .A1(n287), .A2(n283), .ZN(n285) );
  VHSR_CLKNAND2_2 U329 ( .A1(n287), .A2(n283), .ZN(n284) );
  VHSR_NOR2_1 U330 ( .A1(n285), .A2(n286), .ZN(n337) );
  VHSR_AOI22_2 U331 ( .A1(n286), .A2(n285), .B1(n284), .B2(n337), .ZN(n375) );
  VHSR_IAO21_2 U332 ( .A1(n288), .A2(n295), .B(n287), .ZN(n373) );
  VHSR_AD1_1 U333 ( .A(n291), .B(n290), .CI(n289), .CO(n376), .S(n372) );
  VHSR_AD1_1 U334 ( .A(n294), .B(n293), .CI(n292), .CO(n289), .S(n356) );
  VHSR_AOI31_2 U335 ( .A1(n298), .A2(n297), .A3(n296), .B(n295), .ZN(n355) );
  VHSR_AD1_1 U336 ( .A(n301), .B(n300), .CI(n299), .CO(n292), .S(n359) );
  VHSR_AOI22_2 U337 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n303) );
  VHSR_NOR2_1 U338 ( .A1(n303), .A2(n302), .ZN(n358) );
  VHSR_AD1_1 U339 ( .A(n306), .B(n305), .CI(n304), .CO(n299), .S(n362) );
  VHSR_IN_2 U340 ( .I(n320), .ZN(n326) );
  VHSR_AD1_1 U341 ( .A(n378), .B(n308), .CI(n307), .CO(n325), .S(n206) );
  VHSR_AOI22_2 U342 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n324) );
  VHSR_OAI22_2 U343 ( .A1(n312), .A2(n310), .B1(n309), .B2(n313), .ZN(n311) );
  VHSR_OAI31_2 U344 ( .A1(n313), .A2(n312), .A3(n326), .B(n311), .ZN(n334) );
  VHSR_OAI211_2 U345 ( .A1(n320), .A2(n333), .B(a[3]), .C(b[3]), .ZN(n314) );
  VHSR_IN_2 U346 ( .I(n314), .ZN(n365) );
  VHSR_AD1_1 U347 ( .A(n317), .B(n316), .CI(n315), .CO(n304), .S(n364) );
  VHSR_CLKNAND2_2 U348 ( .A1(b[3]), .A2(a[3]), .ZN(n319) );
  VHSR_CLKNAND2_2 U349 ( .A1(n333), .A2(n319), .ZN(n318) );
  VHSR_OAI31_2 U350 ( .A1(n320), .A2(n333), .A3(n319), .B(n318), .ZN(n370) );
  VHSR_AD1_1 U351 ( .A(n323), .B(n322), .CI(n321), .CO(n315), .S(n369) );
  VHSR_AD1_1 U352 ( .A(n326), .B(n325), .CI(n324), .CO(n335), .S(n386) );
  VHSR_NOR2_1 U353 ( .A1(n327), .A2(n389), .ZN(n328) );
  VHSR_AOI32_2 U354 ( .A1(b[0]), .A2(n329), .A3(a[4]), .B1(n328), .B2(n329), 
        .ZN(n385) );
  VHSR_AD1_1 U355 ( .A(n332), .B(n331), .CI(n330), .CO(n321), .S(n367) );
  VHSR_AOI21_2 U356 ( .A1(n335), .A2(n334), .B(n333), .ZN(n366) );
  VHSR_NOR2_1 U357 ( .A1(n337), .A2(n336), .ZN(n349) );
  VHSR_CLKNAND2_2 U358 ( .A1(a[7]), .A2(b[6]), .ZN(n339) );
  VHSR_AOI21_2 U359 ( .A1(a[6]), .A2(b[7]), .B(n339), .ZN(n338) );
  VHSR_AOI31_2 U360 ( .A1(a[6]), .A2(n339), .A3(b[7]), .B(n338), .ZN(n340) );
  VHSR_IN_2 U361 ( .I(n340), .ZN(n341) );
  VHSR_MAOI222_2 U362 ( .A(n344), .B(n342), .C(n341), .ZN(n351) );
  VHSR_OAI21_2 U363 ( .A1(n344), .A2(n343), .B(n351), .ZN(n348) );
  VHSR_CLKXOR2_2 U364 ( .A1(n349), .A2(n348), .Z(n345) );
  VHSR_CLKNAND2_2 U365 ( .A1(n346), .A2(n345), .ZN(n381) );
  VHSR_OAI21_2 U366 ( .A1(n346), .A2(n345), .B(n381), .ZN(n347) );
  VHSR_CLKNAND2_2 U367 ( .A1(a[7]), .A2(b[7]), .ZN(n380) );
  VHSR_NOR2_1 U368 ( .A1(n349), .A2(n348), .ZN(n350) );
  VHSR_AND3_2 U369 ( .A1(n382), .A2(n352), .A3(n381), .Z(n353) );
  VHSR_NOR2_1 U370 ( .A1(n380), .A2(n353), .ZN(product[15]) );
  VHSR_AD1_1 U371 ( .A(n370), .B(n369), .CI(n368), .CO(n363), .S(product[6])
         );
  VHSR_AD1_1 U372 ( .A(n373), .B(n372), .CI(n371), .CO(n374), .S(product[11])
         );
  VHSR_AD1_1 U373 ( .A(n376), .B(n375), .CI(n374), .CO(n346), .S(
        \intadd_0/SUM[7] ) );
  VHSR_AOI222_2 U374 ( .A1(b[2]), .A2(a[0]), .B1(b[1]), .B2(a[1]), .C1(b[0]), 
        .C2(a[2]), .ZN(n377) );
  VHSR_NOR2_1 U375 ( .A1(n380), .A2(n379), .ZN(n383) );
  VHSR_XOR3_2 U376 ( .A1(n383), .A2(n382), .A3(n381), .Z(product[14]) );
  VHSR_AOI21_2 U377 ( .A1(n386), .A2(n385), .B(n384), .ZN(product[4]) );
  VHSR_OAI22_2 U378 ( .A1(n390), .A2(n389), .B1(n388), .B2(n387), .ZN(
        product[1]) );
endmodule

