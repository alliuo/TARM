
module mul8_126 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \mul_ll_ll/out[0] , \intadd_0/SUM[7] , \intadd_0/SUM[3] ,
         \intadd_0/SUM[2] , n205, n206, n207, n208, n209, n210, n211, n212,
         n213, n214, n215, n216, n217, n218, n219, n220, n221, n222, n223,
         n224, n225, n226, n227, n228, n229, n230, n231, n232, n233, n234,
         n235, n236, n237, n238, n239, n240, n241, n242, n243, n244, n245,
         n246, n247, n248, n249, n250, n251, n252, n253, n254, n255, n256,
         n257, n258, n259, n260, n261, n262, n263, n264, n265, n266, n267,
         n268, n269, n270, n271, n272, n273, n274, n275, n276, n277, n278,
         n279, n280, n281, n282, n283, n284, n285, n286, n287, n288, n289,
         n290, n291, n292, n293, n294, n295, n296, n297, n298, n299, n300,
         n301, n302, n303, n304, n305, n306, n307, n308, n309, n310, n311,
         n312, n313, n314, n315, n316, n317, n318, n319, n320, n321, n322,
         n323, n324, n325, n326, n327, n328, n329, n330, n331, n332, n333,
         n334, n335, n336, n337, n338, n339, n340, n341, n342, n343, n344,
         n345, n346, n347, n348, n349, n350, n351, n352, n353, n354, n355,
         n356, n357, n358, n359, n360, n361, n362, n363, n364, n365, n366,
         n367, n368, n369, n370, n371, n372, n373, n374, n375, n376, n377,
         n378, n379, n380, n381, n382, n383, n384, n385, n386, n387, n388,
         n389, n390;
  assign product[0] = \mul_ll_ll/out[0] ;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INAND2_2 U196 ( .A1(n346), .B1(n344), .ZN(n347) );
  VHSR_NOR2_1 U197 ( .A1(n339), .A2(n338), .ZN(n337) );
  VHSR_INOR3_2 U198 ( .A1(n232), .B1(n276), .B2(n315), .ZN(n294) );
  VHSR_NOR2_1 U199 ( .A1(n390), .A2(n389), .ZN(n388) );
  VHSR_INOR2_2 U200 ( .A1(n355), .B1(n354), .ZN(n386) );
  VHSR_INOR2_2 U201 ( .A1(n382), .B1(n381), .ZN(product[2]) );
  VHSR_CLKN_1 U202 ( .I(n351), .ZN(product[13]) );
  VHSR_NOR2_2 U203 ( .A1(n313), .A2(n312), .ZN(n323) );
  VHSR_AD1_1 U204 ( .A(n363), .B(n362), .CI(n361), .CO(n358), .S(product[9])
         );
  VHSR_AD1_1 U205 ( .A(n388), .B(n371), .CI(n370), .CO(n372), .S(product[5])
         );
  VHSR_AD1_1 U206 ( .A(n369), .B(n368), .CI(n367), .CO(n364), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U207 ( .A(n366), .B(n365), .CI(n364), .CO(n361), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U208 ( .A(n360), .B(n359), .CI(n358), .CO(n375), .S(product[10])
         );
  VHSR_PULL0_0 U209 ( .Z(\mul_ll_ll/out[0] ) );
  VHSR_IN_2 U210 ( .I(b[2]), .ZN(n313) );
  VHSR_IN_2 U211 ( .I(a[2]), .ZN(n312) );
  VHSR_IN_2 U212 ( .I(b[0]), .ZN(n216) );
  VHSR_IN_2 U213 ( .I(a[0]), .ZN(n330) );
  VHSR_NOR2_1 U214 ( .A1(n216), .A2(n330), .ZN(n205) );
  VHSR_IN_2 U215 ( .I(b[1]), .ZN(n269) );
  VHSR_NOR2_1 U216 ( .A1(n269), .A2(n330), .ZN(product[1]) );
  VHSR_IN_2 U217 ( .I(a[1]), .ZN(n266) );
  VHSR_NOR2_1 U218 ( .A1(n313), .A2(n266), .ZN(n206) );
  VHSR_AOI22_2 U219 ( .A1(n323), .A2(n205), .B1(product[1]), .B2(n206), .ZN(
        n382) );
  VHSR_AOI22_2 U220 ( .A1(b[1]), .A2(a[2]), .B1(b[0]), .B2(a[3]), .ZN(n311) );
  VHSR_AOI21_2 U221 ( .A1(a[0]), .A2(b[3]), .B(n206), .ZN(n310) );
  VHSR_IN_2 U222 ( .I(n207), .ZN(product[3]) );
  VHSR_AOI22_2 U223 ( .A1(a[7]), .A2(b[2]), .B1(a[6]), .B2(b[3]), .ZN(n243) );
  VHSR_IN_2 U224 ( .I(b[3]), .ZN(n315) );
  VHSR_CLKNAND2_2 U225 ( .A1(b[2]), .A2(a[4]), .ZN(n264) );
  VHSR_IN_2 U226 ( .I(a[5]), .ZN(n280) );
  VHSR_NOR3_2 U227 ( .A1(n315), .A2(n264), .A3(n280), .ZN(n241) );
  VHSR_IN_2 U228 ( .I(a[7]), .ZN(n276) );
  VHSR_NOR2_1 U229 ( .A1(n276), .A2(n269), .ZN(n209) );
  VHSR_AOI211_2 U230 ( .A1(b[2]), .A2(a[4]), .B(n315), .C(n280), .ZN(n210) );
  VHSR_CLKNAND2_2 U231 ( .A1(b[2]), .A2(a[6]), .ZN(n212) );
  VHSR_IN_2 U232 ( .I(n212), .ZN(n208) );
  VHSR_MAOI222_2 U233 ( .A(n209), .B(n210), .C(n208), .ZN(n222) );
  VHSR_AOI21_2 U234 ( .A1(b[1]), .A2(a[7]), .B(n210), .ZN(n213) );
  VHSR_IN_2 U235 ( .I(n222), .ZN(n211) );
  VHSR_AOI21_2 U236 ( .A1(n213), .A2(n212), .B(n211), .ZN(n250) );
  VHSR_CLKNAND2_2 U237 ( .A1(a[6]), .A2(b[1]), .ZN(n219) );
  VHSR_IN_2 U238 ( .I(n219), .ZN(n215) );
  VHSR_IN_2 U239 ( .I(a[4]), .ZN(n281) );
  VHSR_NOR4_2 U240 ( .A1(n281), .A2(n280), .A3(n269), .A4(n216), .ZN(n270) );
  VHSR_AOI22_2 U241 ( .A1(b[2]), .A2(a[5]), .B1(b[3]), .B2(a[4]), .ZN(n214) );
  VHSR_NOR2_1 U242 ( .A1(n241), .A2(n214), .ZN(n217) );
  VHSR_MAOI222_2 U243 ( .A(n215), .B(n270), .C(n217), .ZN(n221) );
  VHSR_OAI21_2 U244 ( .A1(a[7]), .A2(a[6]), .B(b[0]), .ZN(n263) );
  VHSR_OAI211_2 U245 ( .A1(n281), .A2(n216), .B(a[5]), .C(b[1]), .ZN(n262) );
  VHSR_MAOI222_2 U246 ( .A(n264), .B(n263), .C(n262), .ZN(n261) );
  VHSR_NOR2_1 U247 ( .A1(n270), .A2(n217), .ZN(n220) );
  VHSR_IN_2 U248 ( .I(n221), .ZN(n218) );
  VHSR_AOI21_2 U249 ( .A1(n220), .A2(n219), .B(n218), .ZN(n253) );
  VHSR_CLKNAND2_2 U250 ( .A1(n261), .A2(n253), .ZN(n252) );
  VHSR_CLKNAND2_2 U251 ( .A1(n221), .A2(n252), .ZN(n249) );
  VHSR_CLKNAND2_2 U252 ( .A1(n250), .A2(n249), .ZN(n248) );
  VHSR_CLKNAND2_2 U253 ( .A1(n222), .A2(n248), .ZN(n240) );
  VHSR_NOR2_1 U254 ( .A1(n241), .A2(n240), .ZN(n239) );
  VHSR_NOR2_1 U255 ( .A1(n243), .A2(n239), .ZN(n232) );
  VHSR_IN_2 U256 ( .I(b[7]), .ZN(n278) );
  VHSR_IN_2 U257 ( .I(a[3]), .ZN(n316) );
  VHSR_IN_2 U258 ( .I(b[6]), .ZN(n279) );
  VHSR_OAI22_2 U259 ( .A1(n279), .A2(n316), .B1(n278), .B2(n312), .ZN(n238) );
  VHSR_AOI22_2 U260 ( .A1(b[6]), .A2(a[2]), .B1(b[7]), .B2(a[1]), .ZN(n229) );
  VHSR_CLKNAND2_2 U261 ( .A1(b[4]), .A2(a[2]), .ZN(n260) );
  VHSR_NAND3_2 U262 ( .A1(a[3]), .A2(b[5]), .A3(n260), .ZN(n228) );
  VHSR_CLKNAND2_2 U263 ( .A1(b[7]), .A2(a[2]), .ZN(n223) );
  VHSR_CLKNAND2_2 U264 ( .A1(b[6]), .A2(a[1]), .ZN(n225) );
  VHSR_OAI22_2 U265 ( .A1(n229), .A2(n228), .B1(n223), .B2(n225), .ZN(n230) );
  VHSR_IN_2 U266 ( .I(b[4]), .ZN(n331) );
  VHSR_OAI211_2 U267 ( .A1(n331), .A2(n330), .B(b[5]), .C(a[1]), .ZN(n259) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[6]), .A2(a[0]), .ZN(n258) );
  VHSR_MAOI222_2 U269 ( .A(n260), .B(n259), .C(n258), .ZN(n257) );
  VHSR_NAND4_2 U270 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n235) );
  VHSR_IN_2 U271 ( .I(b[5]), .ZN(n275) );
  VHSR_OAI22_2 U272 ( .A1(n331), .A2(n316), .B1(n275), .B2(n312), .ZN(n224) );
  VHSR_AND2_2 U273 ( .A1(n235), .A2(n224), .Z(n227) );
  VHSR_OAI21_2 U274 ( .A1(n278), .A2(n330), .B(n225), .ZN(n226) );
  VHSR_NOR4_2 U275 ( .A1(n331), .A2(n275), .A3(n266), .A4(n330), .ZN(n267) );
  VHSR_AND2_2 U276 ( .A1(n257), .A2(n256), .Z(n255) );
  VHSR_AD1_1 U277 ( .A(n227), .B(n226), .CI(n267), .CO(n244), .S(n256) );
  VHSR_AOI21_2 U278 ( .A1(n229), .A2(n228), .B(n230), .ZN(n247) );
  VHSR_OAI32_2 U279 ( .A1(n230), .A2(n255), .A3(n244), .B1(n247), .B2(n230), 
        .ZN(n236) );
  VHSR_CLKNAND2_2 U280 ( .A1(n236), .A2(n235), .ZN(n234) );
  VHSR_CLKNAND2_2 U281 ( .A1(n238), .A2(n234), .ZN(n233) );
  VHSR_NOR3_2 U282 ( .A1(n278), .A2(n316), .A3(n233), .ZN(n293) );
  VHSR_NOR2_1 U283 ( .A1(n276), .A2(n315), .ZN(n231) );
  VHSR_IAO21_2 U284 ( .A1(n232), .A2(n231), .B(n294), .ZN(n297) );
  VHSR_OAI32_2 U285 ( .A1(n293), .A2(n316), .A3(n278), .B1(n233), .B2(n293), 
        .ZN(n296) );
  VHSR_OAI21_2 U286 ( .A1(n236), .A2(n235), .B(n234), .ZN(n237) );
  VHSR_XNOR2_2 U287 ( .A1(n238), .A2(n237), .ZN(n304) );
  VHSR_AOI21_2 U288 ( .A1(n241), .A2(n240), .B(n239), .ZN(n242) );
  VHSR_XNOR2_2 U289 ( .A1(n243), .A2(n242), .ZN(n303) );
  VHSR_NOR2_1 U290 ( .A1(n255), .A2(n244), .ZN(n246) );
  VHSR_AOI22_2 U291 ( .A1(n255), .A2(n244), .B1(n247), .B2(n246), .ZN(n245) );
  VHSR_OAI21_2 U292 ( .A1(n247), .A2(n246), .B(n245), .ZN(n309) );
  VHSR_OAI21_2 U293 ( .A1(n250), .A2(n249), .B(n248), .ZN(n251) );
  VHSR_IN_2 U294 ( .I(n251), .ZN(n308) );
  VHSR_OAI21_2 U295 ( .A1(n261), .A2(n253), .B(n252), .ZN(n254) );
  VHSR_IN_2 U296 ( .I(n254), .ZN(n320) );
  VHSR_IAO21_2 U297 ( .A1(n257), .A2(n256), .B(n255), .ZN(n319) );
  VHSR_AOI31_2 U298 ( .A1(n260), .A2(n259), .A3(n258), .B(n257), .ZN(n326) );
  VHSR_AOI31_2 U299 ( .A1(n264), .A2(n263), .A3(n262), .B(n261), .ZN(n325) );
  VHSR_CLKNAND2_2 U300 ( .A1(b[5]), .A2(a[0]), .ZN(n265) );
  VHSR_OAI32_2 U301 ( .A1(n267), .A2(n266), .A3(n331), .B1(n265), .B2(n267), 
        .ZN(n336) );
  VHSR_CLKNAND2_2 U302 ( .A1(a[4]), .A2(b[4]), .ZN(n283) );
  VHSR_IN_2 U303 ( .I(n283), .ZN(n365) );
  VHSR_NAND3_2 U304 ( .A1(b[0]), .A2(n365), .A3(a[0]), .ZN(n333) );
  VHSR_IN_2 U305 ( .I(n333), .ZN(n335) );
  VHSR_CLKNAND2_2 U306 ( .A1(a[5]), .A2(b[0]), .ZN(n268) );
  VHSR_OAI32_2 U307 ( .A1(n270), .A2(n269), .A3(n281), .B1(n268), .B2(n270), 
        .ZN(n334) );
  VHSR_CLKNAND2_2 U308 ( .A1(a[6]), .A2(b[6]), .ZN(n356) );
  VHSR_IN_2 U309 ( .I(n356), .ZN(n383) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[6]), .A2(b[4]), .ZN(n301) );
  VHSR_NAND3_2 U311 ( .A1(a[7]), .A2(b[5]), .A3(n301), .ZN(n272) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[4]), .A2(b[6]), .ZN(n300) );
  VHSR_NAND3_2 U313 ( .A1(b[7]), .A2(a[5]), .A3(n300), .ZN(n271) );
  VHSR_CLKNAND2_2 U314 ( .A1(n272), .A2(n271), .ZN(n274) );
  VHSR_MAOI222_2 U315 ( .A(n356), .B(n272), .C(n271), .ZN(n340) );
  VHSR_IN_2 U316 ( .I(n340), .ZN(n273) );
  VHSR_OAI21_2 U317 ( .A1(n383), .A2(n274), .B(n273), .ZN(n289) );
  VHSR_NOR3_2 U318 ( .A1(n280), .A2(n275), .A3(n283), .ZN(n305) );
  VHSR_NOR3_2 U319 ( .A1(n276), .A2(n301), .A3(n275), .ZN(n348) );
  VHSR_AOI22_2 U320 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n277) );
  VHSR_NOR2_1 U321 ( .A1(n348), .A2(n277), .ZN(n285) );
  VHSR_NOR4_2 U322 ( .A1(n281), .A2(n280), .A3(n279), .A4(n278), .ZN(n346) );
  VHSR_AOI22_2 U323 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n282) );
  VHSR_NOR2_1 U324 ( .A1(n346), .A2(n282), .ZN(n284) );
  VHSR_NAND3_2 U325 ( .A1(b[5]), .A2(a[5]), .A3(n283), .ZN(n299) );
  VHSR_MAOI222_2 U326 ( .A(n301), .B(n300), .C(n299), .ZN(n298) );
  VHSR_AND2_2 U327 ( .A1(n291), .A2(n298), .Z(n290) );
  VHSR_AD1_1 U328 ( .A(n305), .B(n285), .CI(n284), .CO(n286), .S(n291) );
  VHSR_NOR2_1 U329 ( .A1(n290), .A2(n286), .ZN(n288) );
  VHSR_CLKNAND2_2 U330 ( .A1(n290), .A2(n286), .ZN(n287) );
  VHSR_NOR2_1 U331 ( .A1(n288), .A2(n289), .ZN(n341) );
  VHSR_AOI22_2 U332 ( .A1(n289), .A2(n288), .B1(n287), .B2(n341), .ZN(n379) );
  VHSR_IAO21_2 U333 ( .A1(n291), .A2(n298), .B(n290), .ZN(n377) );
  VHSR_AD1_1 U334 ( .A(n294), .B(n293), .CI(n292), .CO(n380), .S(n376) );
  VHSR_AD1_1 U335 ( .A(n297), .B(n296), .CI(n295), .CO(n292), .S(n360) );
  VHSR_AOI31_2 U336 ( .A1(n301), .A2(n300), .A3(n299), .B(n298), .ZN(n359) );
  VHSR_AD1_1 U337 ( .A(n304), .B(n303), .CI(n302), .CO(n295), .S(n363) );
  VHSR_AOI22_2 U338 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n306) );
  VHSR_NOR2_1 U339 ( .A1(n306), .A2(n305), .ZN(n362) );
  VHSR_AD1_1 U340 ( .A(n309), .B(n308), .CI(n307), .CO(n302), .S(n366) );
  VHSR_IN_2 U341 ( .I(n323), .ZN(n329) );
  VHSR_AD1_1 U342 ( .A(n382), .B(n311), .CI(n310), .CO(n328), .S(n207) );
  VHSR_AOI22_2 U343 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n327) );
  VHSR_OAI22_2 U344 ( .A1(n313), .A2(n316), .B1(n315), .B2(n312), .ZN(n314) );
  VHSR_OAI31_2 U345 ( .A1(n316), .A2(n315), .A3(n329), .B(n314), .ZN(n338) );
  VHSR_OAI211_2 U346 ( .A1(n323), .A2(n337), .B(a[3]), .C(b[3]), .ZN(n317) );
  VHSR_IN_2 U347 ( .I(n317), .ZN(n369) );
  VHSR_AD1_1 U348 ( .A(n320), .B(n319), .CI(n318), .CO(n307), .S(n368) );
  VHSR_CLKNAND2_2 U349 ( .A1(b[3]), .A2(a[3]), .ZN(n322) );
  VHSR_CLKNAND2_2 U350 ( .A1(n337), .A2(n322), .ZN(n321) );
  VHSR_OAI31_2 U351 ( .A1(n323), .A2(n337), .A3(n322), .B(n321), .ZN(n374) );
  VHSR_AD1_1 U352 ( .A(n326), .B(n325), .CI(n324), .CO(n318), .S(n373) );
  VHSR_AD1_1 U353 ( .A(n329), .B(n328), .CI(n327), .CO(n339), .S(n390) );
  VHSR_NOR2_1 U354 ( .A1(n331), .A2(n330), .ZN(n332) );
  VHSR_AOI32_2 U355 ( .A1(b[0]), .A2(n333), .A3(a[4]), .B1(n332), .B2(n333), 
        .ZN(n389) );
  VHSR_AD1_1 U356 ( .A(n336), .B(n335), .CI(n334), .CO(n324), .S(n371) );
  VHSR_AOI21_2 U357 ( .A1(n339), .A2(n338), .B(n337), .ZN(n370) );
  VHSR_NOR2_1 U358 ( .A1(n341), .A2(n340), .ZN(n353) );
  VHSR_CLKNAND2_2 U359 ( .A1(a[6]), .A2(b[7]), .ZN(n343) );
  VHSR_AOI21_2 U360 ( .A1(a[7]), .A2(b[6]), .B(n343), .ZN(n342) );
  VHSR_AOI31_2 U361 ( .A1(a[7]), .A2(n343), .A3(b[6]), .B(n342), .ZN(n344) );
  VHSR_IN_2 U362 ( .I(n344), .ZN(n345) );
  VHSR_MAOI222_2 U363 ( .A(n348), .B(n346), .C(n345), .ZN(n355) );
  VHSR_OAI21_2 U364 ( .A1(n348), .A2(n347), .B(n355), .ZN(n352) );
  VHSR_CLKXOR2_2 U365 ( .A1(n353), .A2(n352), .Z(n349) );
  VHSR_CLKNAND2_2 U366 ( .A1(n350), .A2(n349), .ZN(n385) );
  VHSR_OAI21_2 U367 ( .A1(n350), .A2(n349), .B(n385), .ZN(n351) );
  VHSR_CLKNAND2_2 U368 ( .A1(a[7]), .A2(b[7]), .ZN(n384) );
  VHSR_NOR2_1 U369 ( .A1(n353), .A2(n352), .ZN(n354) );
  VHSR_AND3_2 U370 ( .A1(n386), .A2(n356), .A3(n385), .Z(n357) );
  VHSR_NOR2_1 U371 ( .A1(n384), .A2(n357), .ZN(product[15]) );
  VHSR_AD1_1 U372 ( .A(n374), .B(n373), .CI(n372), .CO(n367), .S(product[6])
         );
  VHSR_AD1_1 U373 ( .A(n377), .B(n376), .CI(n375), .CO(n378), .S(product[11])
         );
  VHSR_AD1_1 U374 ( .A(n380), .B(n379), .CI(n378), .CO(n350), .S(
        \intadd_0/SUM[7] ) );
  VHSR_AOI222_2 U375 ( .A1(b[2]), .A2(a[0]), .B1(b[1]), .B2(a[1]), .C1(b[0]), 
        .C2(a[2]), .ZN(n381) );
  VHSR_NOR2_1 U376 ( .A1(n384), .A2(n383), .ZN(n387) );
  VHSR_XOR3_2 U377 ( .A1(n387), .A2(n386), .A3(n385), .Z(product[14]) );
  VHSR_AOI21_2 U378 ( .A1(n390), .A2(n389), .B(n388), .ZN(product[4]) );
endmodule

