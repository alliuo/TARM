
module mul8_55 ( a, b, product );
  input [7:0] a;
  input [7:0] b;
  output [15:0] product;
  wire   \intadd_0/SUM[3] , \intadd_0/SUM[2] , n210, n211, n212, n213, n214,
         n215, n216, n217, n218, n219, n220, n221, n222, n223, n224, n225,
         n226, n227, n228, n229, n230, n231, n232, n233, n234, n235, n236,
         n237, n238, n239, n240, n241, n242, n243, n244, n245, n246, n247,
         n248, n249, n250, n251, n252, n253, n254, n255, n256, n257, n258,
         n259, n260, n261, n262, n263, n264, n265, n266, n267, n268, n269,
         n270, n271, n272, n273, n274, n275, n276, n277, n278, n279, n280,
         n281, n282, n283, n284, n285, n286, n287, n288, n289, n290, n291,
         n292, n293, n294, n295, n296, n297, n298, n299, n300, n301, n302,
         n303, n304, n305, n306, n307, n308, n309, n310, n311, n312, n313,
         n314, n315, n316, n317, n318, n319, n320, n321, n322, n323, n324,
         n325, n326, n327, n328, n329, n330, n331, n332, n333, n334, n335,
         n336, n337, n338, n339, n340, n341, n342, n343, n344, n345, n346,
         n347, n348, n349, n350, n351, n352, n353, n354, n355, n356, n357,
         n358, n359, n360, n361, n362, n363, n364, n365, n366, n367, n368,
         n369, n370, n371, n372, n373, n374, n375, n376, n377, n378, n379,
         n380, n381, n382, n383, n384, n385, n386, n387, n388, n389, n390,
         n391, n392, n393, n394, n395, n396, n397, n398;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U202 ( .A1(n255), .B1(n276), .ZN(n263) );
  VHSR_IOA21_2 U203 ( .A1(n254), .A2(n253), .B(n255), .ZN(n279) );
  VHSR_NOR2_1 U204 ( .A1(n396), .A2(n330), .ZN(n341) );
  VHSR_INOR3_2 U205 ( .A1(n258), .B1(n329), .B2(n256), .ZN(n307) );
  VHSR_IOA21_2 U206 ( .A1(n388), .A2(n387), .B(n386), .ZN(n390) );
  VHSR_IN_2 U207 ( .I(n352), .ZN(product[15]) );
  VHSR_MOAI22_1 U208 ( .A1(n330), .A2(n248), .B1(a[3]), .B2(b[4]), .ZN(n249)
         );
  VHSR_NOR2_2 U209 ( .A1(n239), .A2(n250), .ZN(n383) );
  VHSR_AD1_1 U210 ( .A(n364), .B(n363), .CI(n362), .CO(n359), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U211 ( .A(n358), .B(n357), .CI(n356), .CO(n353), .S(product[11])
         );
  VHSR_AD1_1 U212 ( .A(n371), .B(n398), .CI(n370), .CO(n344), .S(product[3])
         );
  VHSR_AD1_1 U213 ( .A(n389), .B(n369), .CI(n368), .CO(n372), .S(product[5])
         );
  VHSR_AD1_1 U214 ( .A(n367), .B(n366), .CI(n365), .CO(n362), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U215 ( .A(n361), .B(n360), .CI(n359), .CO(n375), .S(product[9])
         );
  VHSR_AD1_1 U216 ( .A(n355), .B(n354), .CI(n353), .CO(n378), .S(product[12])
         );
  VHSR_CLKNAND2_2 U217 ( .A1(a[7]), .A2(b[7]), .ZN(n381) );
  VHSR_IN_2 U218 ( .I(n381), .ZN(n351) );
  VHSR_IN_2 U219 ( .I(a[6]), .ZN(n239) );
  VHSR_IN_2 U220 ( .I(b[6]), .ZN(n250) );
  VHSR_AOI22_2 U221 ( .A1(a[6]), .A2(b[7]), .B1(a[7]), .B2(b[6]), .ZN(n210) );
  VHSR_AOI21_2 U222 ( .A1(n351), .A2(n383), .B(n210), .ZN(n229) );
  VHSR_IN_2 U223 ( .I(b[5]), .ZN(n248) );
  VHSR_IN_2 U224 ( .I(a[7]), .ZN(n256) );
  VHSR_CLKNAND2_2 U225 ( .A1(a[6]), .A2(b[4]), .ZN(n313) );
  VHSR_NAND4_2 U226 ( .A1(a[5]), .A2(a[4]), .A3(b[7]), .A4(b[6]), .ZN(n217) );
  VHSR_OAI31_2 U227 ( .A1(n248), .A2(n256), .A3(n313), .B(n217), .ZN(n228) );
  VHSR_IN_2 U228 ( .I(a[5]), .ZN(n236) );
  VHSR_IN_2 U229 ( .I(b[7]), .ZN(n260) );
  VHSR_AOI211_2 U230 ( .A1(a[4]), .A2(b[6]), .B(n236), .C(n260), .ZN(n212) );
  VHSR_AOI211_2 U231 ( .A1(a[6]), .A2(b[4]), .B(n256), .C(n248), .ZN(n211) );
  VHSR_MAOI222_2 U232 ( .A(n212), .B(n383), .C(n211), .ZN(n226) );
  VHSR_AOI31_2 U233 ( .A1(b[5]), .A2(a[7]), .A3(n313), .B(n383), .ZN(n215) );
  VHSR_IN_2 U234 ( .I(n212), .ZN(n214) );
  VHSR_IN_2 U235 ( .I(n226), .ZN(n213) );
  VHSR_AOI21_2 U236 ( .A1(n215), .A2(n214), .B(n213), .ZN(n300) );
  VHSR_IN_2 U237 ( .I(a[4]), .ZN(n235) );
  VHSR_NOR2_1 U238 ( .A1(n235), .A2(n260), .ZN(n216) );
  VHSR_AOI32_2 U239 ( .A1(n217), .A2(b[6]), .A3(a[5]), .B1(n216), .B2(n217), 
        .ZN(n223) );
  VHSR_IN_2 U240 ( .I(n223), .ZN(n220) );
  VHSR_CLKNAND2_2 U241 ( .A1(a[4]), .A2(b[4]), .ZN(n297) );
  VHSR_NOR3_2 U242 ( .A1(n236), .A2(n248), .A3(n297), .ZN(n319) );
  VHSR_IN_2 U243 ( .I(n313), .ZN(n219) );
  VHSR_AOI22_2 U244 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n218) );
  VHSR_AOI31_2 U245 ( .A1(b[5]), .A2(a[7]), .A3(n219), .B(n218), .ZN(n221) );
  VHSR_MAOI222_2 U246 ( .A(n220), .B(n319), .C(n221), .ZN(n225) );
  VHSR_CLKNAND2_2 U247 ( .A1(a[4]), .A2(b[6]), .ZN(n314) );
  VHSR_NAND3_2 U248 ( .A1(b[5]), .A2(a[5]), .A3(n297), .ZN(n312) );
  VHSR_MAOI222_2 U249 ( .A(n314), .B(n313), .C(n312), .ZN(n311) );
  VHSR_NOR2_1 U250 ( .A1(n319), .A2(n221), .ZN(n224) );
  VHSR_IN_2 U251 ( .I(n225), .ZN(n222) );
  VHSR_AOI21_2 U252 ( .A1(n224), .A2(n223), .B(n222), .ZN(n303) );
  VHSR_CLKNAND2_2 U253 ( .A1(n311), .A2(n303), .ZN(n302) );
  VHSR_CLKNAND2_2 U254 ( .A1(n225), .A2(n302), .ZN(n299) );
  VHSR_CLKNAND2_2 U255 ( .A1(n300), .A2(n299), .ZN(n298) );
  VHSR_CLKNAND2_2 U256 ( .A1(n226), .A2(n298), .ZN(n227) );
  VHSR_AD1_1 U257 ( .A(n229), .B(n228), .CI(n227), .CO(n382), .S(n379) );
  VHSR_AOI22_2 U258 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n270) );
  VHSR_IN_2 U259 ( .I(b[3]), .ZN(n329) );
  VHSR_IN_2 U260 ( .I(b[2]), .ZN(n396) );
  VHSR_NOR4_2 U261 ( .A1(n329), .A2(n396), .A3(n236), .A4(n235), .ZN(n268) );
  VHSR_IN_2 U262 ( .I(b[1]), .ZN(n394) );
  VHSR_NOR2_1 U263 ( .A1(n256), .A2(n394), .ZN(n231) );
  VHSR_NOR2_1 U264 ( .A1(n239), .A2(n396), .ZN(n230) );
  VHSR_AOI211_2 U265 ( .A1(b[2]), .A2(a[4]), .B(n329), .C(n236), .ZN(n232) );
  VHSR_MAOI222_2 U266 ( .A(n231), .B(n230), .C(n232), .ZN(n245) );
  VHSR_OAI22_2 U267 ( .A1(n239), .A2(n396), .B1(n256), .B2(n394), .ZN(n233) );
  VHSR_OAI21_2 U268 ( .A1(n233), .A2(n232), .B(n245), .ZN(n234) );
  VHSR_IN_2 U269 ( .I(n234), .ZN(n273) );
  VHSR_IN_2 U270 ( .I(b[0]), .ZN(n393) );
  VHSR_NOR4_2 U271 ( .A1(n236), .A2(n235), .A3(n394), .A4(n393), .ZN(n296) );
  VHSR_CLKNAND2_2 U272 ( .A1(b[2]), .A2(a[5]), .ZN(n238) );
  VHSR_CLKNAND2_2 U273 ( .A1(b[3]), .A2(a[4]), .ZN(n237) );
  VHSR_AOI21_2 U274 ( .A1(n238), .A2(n237), .B(n268), .ZN(n240) );
  VHSR_OAI22_2 U275 ( .A1(n239), .A2(n394), .B1(n256), .B2(n393), .ZN(n241) );
  VHSR_MAOI222_2 U276 ( .A(n296), .B(n240), .C(n241), .ZN(n244) );
  VHSR_CLKNAND2_2 U277 ( .A1(b[2]), .A2(a[4]), .ZN(n292) );
  VHSR_CLKNAND2_2 U278 ( .A1(a[4]), .A2(b[0]), .ZN(n388) );
  VHSR_NAND3_2 U279 ( .A1(b[1]), .A2(a[5]), .A3(n388), .ZN(n291) );
  VHSR_CLKNAND2_2 U280 ( .A1(a[6]), .A2(b[0]), .ZN(n290) );
  VHSR_MAOI222_2 U281 ( .A(n292), .B(n291), .C(n290), .ZN(n289) );
  VHSR_OR2_2 U282 ( .A1(n296), .A2(n240), .Z(n242) );
  VHSR_OAI21_2 U283 ( .A1(n242), .A2(n241), .B(n244), .ZN(n243) );
  VHSR_IN_2 U284 ( .I(n243), .ZN(n283) );
  VHSR_CLKNAND2_2 U285 ( .A1(n289), .A2(n283), .ZN(n282) );
  VHSR_CLKNAND2_2 U286 ( .A1(n244), .A2(n282), .ZN(n272) );
  VHSR_CLKNAND2_2 U287 ( .A1(n273), .A2(n272), .ZN(n271) );
  VHSR_CLKNAND2_2 U288 ( .A1(n245), .A2(n271), .ZN(n267) );
  VHSR_NOR2_1 U289 ( .A1(n268), .A2(n267), .ZN(n266) );
  VHSR_NOR2_1 U290 ( .A1(n270), .A2(n266), .ZN(n258) );
  VHSR_IN_2 U291 ( .I(a[3]), .ZN(n332) );
  VHSR_IN_2 U292 ( .I(a[2]), .ZN(n330) );
  VHSR_OAI22_2 U293 ( .A1(n330), .A2(n260), .B1(n332), .B2(n250), .ZN(n265) );
  VHSR_NOR2_1 U294 ( .A1(n330), .A2(n260), .ZN(n247) );
  VHSR_IN_2 U295 ( .I(a[1]), .ZN(n392) );
  VHSR_NOR2_1 U296 ( .A1(n250), .A2(n392), .ZN(n246) );
  VHSR_CLKNAND2_2 U297 ( .A1(a[2]), .A2(b[4]), .ZN(n288) );
  VHSR_NAND3_2 U298 ( .A1(a[3]), .A2(b[5]), .A3(n288), .ZN(n253) );
  VHSR_AOI22_2 U299 ( .A1(a[2]), .A2(b[6]), .B1(b[7]), .B2(a[1]), .ZN(n254) );
  VHSR_IAO22_2 U300 ( .B1(n247), .B2(n246), .A1(n253), .A2(n254), .ZN(n255) );
  VHSR_CLKNAND2_2 U301 ( .A1(b[4]), .A2(a[0]), .ZN(n387) );
  VHSR_NAND3_2 U302 ( .A1(a[1]), .A2(b[5]), .A3(n387), .ZN(n287) );
  VHSR_CLKNAND2_2 U303 ( .A1(b[6]), .A2(a[0]), .ZN(n286) );
  VHSR_MAOI222_2 U304 ( .A(n288), .B(n287), .C(n286), .ZN(n285) );
  VHSR_NOR3_2 U305 ( .A1(n248), .A2(n392), .A3(n387), .ZN(n293) );
  VHSR_NAND4_2 U306 ( .A1(a[2]), .A2(a[3]), .A3(b[4]), .A4(b[5]), .ZN(n262) );
  VHSR_AND2_2 U307 ( .A1(n262), .A2(n249), .Z(n252) );
  VHSR_IN_2 U308 ( .I(a[0]), .ZN(n397) );
  VHSR_OAI22_2 U309 ( .A1(n260), .A2(n397), .B1(n250), .B2(n392), .ZN(n251) );
  VHSR_AND2_2 U310 ( .A1(n285), .A2(n281), .Z(n280) );
  VHSR_AD1_1 U311 ( .A(n293), .B(n252), .CI(n251), .CO(n275), .S(n281) );
  VHSR_NOR2_1 U312 ( .A1(n280), .A2(n275), .ZN(n278) );
  VHSR_NOR2_1 U313 ( .A1(n278), .A2(n279), .ZN(n276) );
  VHSR_CLKNAND2_2 U314 ( .A1(n263), .A2(n262), .ZN(n261) );
  VHSR_CLKNAND2_2 U315 ( .A1(n265), .A2(n261), .ZN(n259) );
  VHSR_NOR3_2 U316 ( .A1(n260), .A2(n332), .A3(n259), .ZN(n306) );
  VHSR_NOR2_1 U317 ( .A1(n329), .A2(n256), .ZN(n257) );
  VHSR_IAO21_2 U318 ( .A1(n258), .A2(n257), .B(n307), .ZN(n310) );
  VHSR_OAI32_2 U319 ( .A1(n306), .A2(n332), .A3(n260), .B1(n259), .B2(n306), 
        .ZN(n309) );
  VHSR_OAI21_2 U320 ( .A1(n263), .A2(n262), .B(n261), .ZN(n264) );
  VHSR_XNOR2_2 U321 ( .A1(n265), .A2(n264), .ZN(n317) );
  VHSR_AOI21_2 U322 ( .A1(n268), .A2(n267), .B(n266), .ZN(n269) );
  VHSR_XNOR2_2 U323 ( .A1(n270), .A2(n269), .ZN(n316) );
  VHSR_OAI21_2 U324 ( .A1(n273), .A2(n272), .B(n271), .ZN(n274) );
  VHSR_IN_2 U325 ( .I(n274), .ZN(n322) );
  VHSR_CLKNAND2_2 U326 ( .A1(n280), .A2(n275), .ZN(n277) );
  VHSR_AOI22_2 U327 ( .A1(n279), .A2(n278), .B1(n277), .B2(n276), .ZN(n321) );
  VHSR_IAO21_2 U328 ( .A1(n285), .A2(n281), .B(n280), .ZN(n325) );
  VHSR_OAI21_2 U329 ( .A1(n289), .A2(n283), .B(n282), .ZN(n284) );
  VHSR_IN_2 U330 ( .I(n284), .ZN(n324) );
  VHSR_AOI31_2 U331 ( .A1(n288), .A2(n287), .A3(n286), .B(n285), .ZN(n338) );
  VHSR_AOI31_2 U332 ( .A1(n292), .A2(n291), .A3(n290), .B(n289), .ZN(n337) );
  VHSR_AOI22_2 U333 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n294) );
  VHSR_NOR2_1 U334 ( .A1(n294), .A2(n293), .ZN(n350) );
  VHSR_AOI22_2 U335 ( .A1(a[5]), .A2(b[0]), .B1(a[4]), .B2(b[1]), .ZN(n295) );
  VHSR_NOR2_1 U336 ( .A1(n296), .A2(n295), .ZN(n349) );
  VHSR_IN_2 U337 ( .I(n297), .ZN(n363) );
  VHSR_NOR2_1 U338 ( .A1(n393), .A2(n397), .ZN(product[0]) );
  VHSR_CLKNAND2_2 U339 ( .A1(n363), .A2(product[0]), .ZN(n386) );
  VHSR_IN_2 U340 ( .I(n386), .ZN(n348) );
  VHSR_OAI21_2 U341 ( .A1(n300), .A2(n299), .B(n298), .ZN(n301) );
  VHSR_IN_2 U342 ( .I(n301), .ZN(n354) );
  VHSR_OAI21_2 U343 ( .A1(n311), .A2(n303), .B(n302), .ZN(n304) );
  VHSR_IN_2 U344 ( .I(n304), .ZN(n358) );
  VHSR_AD1_1 U345 ( .A(n307), .B(n306), .CI(n305), .CO(n355), .S(n357) );
  VHSR_AD1_1 U346 ( .A(n310), .B(n309), .CI(n308), .CO(n305), .S(n377) );
  VHSR_AOI31_2 U347 ( .A1(n314), .A2(n313), .A3(n312), .B(n311), .ZN(n376) );
  VHSR_AD1_1 U348 ( .A(n317), .B(n316), .CI(n315), .CO(n308), .S(n361) );
  VHSR_AOI22_2 U349 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n318) );
  VHSR_NOR2_1 U350 ( .A1(n319), .A2(n318), .ZN(n360) );
  VHSR_AD1_1 U351 ( .A(n322), .B(n321), .CI(n320), .CO(n315), .S(n364) );
  VHSR_AD1_1 U352 ( .A(n325), .B(n324), .CI(n323), .CO(n320), .S(n367) );
  VHSR_AOI22_2 U353 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n326) );
  VHSR_AOI31_2 U354 ( .A1(a[3]), .A2(b[3]), .A3(n341), .B(n326), .ZN(n347) );
  VHSR_NOR2_1 U355 ( .A1(n329), .A2(n392), .ZN(n328) );
  VHSR_NOR2_1 U356 ( .A1(n394), .A2(n332), .ZN(n327) );
  VHSR_MAOI222_2 U357 ( .A(n341), .B(n328), .C(n327), .ZN(n334) );
  VHSR_OAI22_2 U358 ( .A1(n329), .A2(n397), .B1(n396), .B2(n392), .ZN(n371) );
  VHSR_AOI22_2 U359 ( .A1(b[1]), .A2(a[1]), .B1(b[0]), .B2(a[2]), .ZN(n395) );
  VHSR_NOR3_2 U360 ( .A1(n395), .A2(n397), .A3(n396), .ZN(n398) );
  VHSR_OAI22_2 U361 ( .A1(n394), .A2(n330), .B1(n393), .B2(n332), .ZN(n370) );
  VHSR_IN_2 U362 ( .I(n334), .ZN(n333) );
  VHSR_AOI21_2 U363 ( .A1(a[1]), .A2(b[3]), .B(n341), .ZN(n331) );
  VHSR_OAI32_2 U364 ( .A1(n333), .A2(n332), .A3(n394), .B1(n331), .B2(n333), 
        .ZN(n343) );
  VHSR_CLKNAND2_2 U365 ( .A1(n344), .A2(n343), .ZN(n342) );
  VHSR_CLKNAND2_2 U366 ( .A1(n334), .A2(n342), .ZN(n346) );
  VHSR_AND2_2 U367 ( .A1(n347), .A2(n346), .Z(n345) );
  VHSR_OAI211_2 U368 ( .A1(n341), .A2(n345), .B(a[3]), .C(b[3]), .ZN(n335) );
  VHSR_IN_2 U369 ( .I(n335), .ZN(n366) );
  VHSR_AD1_1 U370 ( .A(n338), .B(n337), .CI(n336), .CO(n323), .S(n374) );
  VHSR_CLKNAND2_2 U371 ( .A1(b[3]), .A2(a[3]), .ZN(n340) );
  VHSR_CLKNAND2_2 U372 ( .A1(n345), .A2(n340), .ZN(n339) );
  VHSR_OAI31_2 U373 ( .A1(n341), .A2(n345), .A3(n340), .B(n339), .ZN(n373) );
  VHSR_OAI21_2 U374 ( .A1(n344), .A2(n343), .B(n342), .ZN(n391) );
  VHSR_AOI211_2 U375 ( .A1(n388), .A2(n387), .B(n348), .C(n391), .ZN(n389) );
  VHSR_IAO21_2 U376 ( .A1(n347), .A2(n346), .B(n345), .ZN(n369) );
  VHSR_AD1_1 U377 ( .A(n350), .B(n349), .CI(n348), .CO(n336), .S(n368) );
  VHSR_AND2_2 U378 ( .A1(n379), .A2(n378), .Z(n385) );
  VHSR_OAI31_2 U379 ( .A1(n382), .A2(n385), .A3(n383), .B(n351), .ZN(n352) );
  VHSR_AD1_1 U380 ( .A(n374), .B(n373), .CI(n372), .CO(n365), .S(product[6])
         );
  VHSR_AD1_1 U381 ( .A(n377), .B(n376), .CI(n375), .CO(n356), .S(product[10])
         );
  VHSR_IAO21_2 U382 ( .A1(n379), .A2(n378), .B(n385), .ZN(product[13]) );
  VHSR_OAI21_2 U383 ( .A1(n383), .A2(n381), .B(n382), .ZN(n380) );
  VHSR_OAI31_2 U384 ( .A1(n383), .A2(n382), .A3(n381), .B(n380), .ZN(n384) );
  VHSR_CLKXOR2_2 U385 ( .A1(n385), .A2(n384), .Z(product[14]) );
  VHSR_AOI21_2 U386 ( .A1(n391), .A2(n390), .B(n389), .ZN(product[4]) );
  VHSR_OAI22_2 U387 ( .A1(n394), .A2(n397), .B1(n393), .B2(n392), .ZN(
        product[1]) );
  VHSR_OAI32_2 U388 ( .A1(n398), .A2(n397), .A3(n396), .B1(n395), .B2(n398), 
        .ZN(product[2]) );
endmodule

