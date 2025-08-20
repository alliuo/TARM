
module mul8_64 ( a, b, product );
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
         n391, n392, n393, n394, n395, n396;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U202 ( .A1(a[6]), .B1(n392), .ZN(n241) );
  VHSR_INOR2_2 U203 ( .A1(n255), .B1(n276), .ZN(n263) );
  VHSR_NOR2_1 U204 ( .A1(n278), .A2(n279), .ZN(n276) );
  VHSR_IOA21_2 U205 ( .A1(n254), .A2(n253), .B(n255), .ZN(n279) );
  VHSR_NOR2_1 U206 ( .A1(n270), .A2(n266), .ZN(n258) );
  VHSR_INAND2_2 U207 ( .A1(n327), .B1(n339), .ZN(n343) );
  VHSR_INAND3_2 U208 ( .A1(n359), .B1(b[5]), .B2(a[5]), .ZN(n311) );
  VHSR_IN_2 U209 ( .I(n375), .ZN(n377) );
  VHSR_NOR2_1 U210 ( .A1(n318), .A2(n317), .ZN(n355) );
  VHSR_IN_2 U211 ( .I(n347), .ZN(product[15]) );
  VHSR_AD1_2 U212 ( .A(n306), .B(n305), .CI(n304), .CO(n350), .S(n352) );
  VHSR_NOR2_2 U213 ( .A1(n268), .A2(n267), .ZN(n266) );
  VHSR_NOR2_2 U214 ( .A1(n218), .A2(n217), .ZN(n226) );
  VHSR_MOAI22_1 U215 ( .A1(n325), .A2(n250), .B1(a[3]), .B2(b[4]), .ZN(n248)
         );
  VHSR_MOAI22_1 U216 ( .A1(n323), .A2(n391), .B1(b[2]), .B2(a[1]), .ZN(n366)
         );
  VHSR_CLKN_1 U217 ( .I(n334), .ZN(n338) );
  VHSR_NOR2_2 U218 ( .A1(n256), .A2(n260), .ZN(n375) );
  VHSR_CLKN_1 U219 ( .I(n312), .ZN(n220) );
  VHSR_NOR2_2 U220 ( .A1(n384), .A2(n383), .ZN(n382) );
  VHSR_AD1_1 U221 ( .A(n359), .B(n358), .CI(n357), .CO(n354), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U222 ( .A(n353), .B(n352), .CI(n351), .CO(n348), .S(product[11])
         );
  VHSR_AD1_1 U223 ( .A(n366), .B(n393), .CI(n365), .CO(n341), .S(product[3])
         );
  VHSR_AD1_1 U224 ( .A(n386), .B(n364), .CI(n363), .CO(n367), .S(product[5])
         );
  VHSR_AD1_1 U225 ( .A(n362), .B(n361), .CI(n360), .CO(n357), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U226 ( .A(n356), .B(n355), .CI(n354), .CO(n370), .S(product[9])
         );
  VHSR_AD1_1 U227 ( .A(n350), .B(n349), .CI(n348), .CO(n373), .S(product[12])
         );
  VHSR_IN_2 U228 ( .I(b[0]), .ZN(n390) );
  VHSR_IN_2 U229 ( .I(a[0]), .ZN(n391) );
  VHSR_NOR2_1 U230 ( .A1(n390), .A2(n391), .ZN(product[0]) );
  VHSR_IN_2 U231 ( .I(a[7]), .ZN(n256) );
  VHSR_IN_2 U232 ( .I(b[7]), .ZN(n260) );
  VHSR_CLKNAND2_2 U233 ( .A1(a[6]), .A2(b[6]), .ZN(n213) );
  VHSR_IN_2 U234 ( .I(n213), .ZN(n379) );
  VHSR_AOI22_2 U235 ( .A1(a[6]), .A2(b[7]), .B1(a[7]), .B2(b[6]), .ZN(n210) );
  VHSR_AOI21_2 U236 ( .A1(n375), .A2(n379), .B(n210), .ZN(n231) );
  VHSR_IN_2 U237 ( .I(b[5]), .ZN(n250) );
  VHSR_CLKNAND2_2 U238 ( .A1(a[6]), .A2(b[4]), .ZN(n312) );
  VHSR_IN_2 U239 ( .I(a[5]), .ZN(n238) );
  VHSR_CLKNAND2_2 U240 ( .A1(a[4]), .A2(b[6]), .ZN(n313) );
  VHSR_NOR3_2 U241 ( .A1(n238), .A2(n260), .A3(n313), .ZN(n218) );
  VHSR_IN_2 U242 ( .I(n218), .ZN(n211) );
  VHSR_OAI31_2 U243 ( .A1(n250), .A2(n256), .A3(n312), .B(n211), .ZN(n230) );
  VHSR_AOI211_2 U244 ( .A1(a[4]), .A2(b[6]), .B(n238), .C(n260), .ZN(n214) );
  VHSR_AOI211_2 U245 ( .A1(a[6]), .A2(b[4]), .B(n256), .C(n250), .ZN(n212) );
  VHSR_MAOI222_2 U246 ( .A(n214), .B(n379), .C(n212), .ZN(n228) );
  VHSR_OAI31_2 U247 ( .A1(n250), .A2(n256), .A3(n220), .B(n213), .ZN(n215) );
  VHSR_OAI21_2 U248 ( .A1(n215), .A2(n214), .B(n228), .ZN(n216) );
  VHSR_IN_2 U249 ( .I(n216), .ZN(n299) );
  VHSR_AOI22_2 U250 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n217) );
  VHSR_AND2_2 U251 ( .A1(a[4]), .A2(b[4]), .Z(n359) );
  VHSR_NAND3_2 U252 ( .A1(a[5]), .A2(b[5]), .A3(n359), .ZN(n222) );
  VHSR_IN_2 U253 ( .I(n222), .ZN(n318) );
  VHSR_AOI22_2 U254 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n219) );
  VHSR_AOI31_2 U255 ( .A1(b[5]), .A2(a[7]), .A3(n220), .B(n219), .ZN(n221) );
  VHSR_MAOI222_2 U256 ( .A(n226), .B(n318), .C(n221), .ZN(n227) );
  VHSR_MAOI222_2 U257 ( .A(n313), .B(n312), .C(n311), .ZN(n310) );
  VHSR_IN_2 U258 ( .I(n221), .ZN(n223) );
  VHSR_CLKNAND2_2 U259 ( .A1(n222), .A2(n223), .ZN(n225) );
  VHSR_OAI22_2 U260 ( .A1(n226), .A2(n225), .B1(n223), .B2(n222), .ZN(n224) );
  VHSR_AOI21_2 U261 ( .A1(n226), .A2(n225), .B(n224), .ZN(n302) );
  VHSR_CLKNAND2_2 U262 ( .A1(n310), .A2(n302), .ZN(n301) );
  VHSR_CLKNAND2_2 U263 ( .A1(n227), .A2(n301), .ZN(n298) );
  VHSR_CLKNAND2_2 U264 ( .A1(n299), .A2(n298), .ZN(n297) );
  VHSR_CLKNAND2_2 U265 ( .A1(n228), .A2(n297), .ZN(n229) );
  VHSR_AD1_1 U266 ( .A(n231), .B(n230), .CI(n229), .CO(n378), .S(n374) );
  VHSR_AOI22_2 U267 ( .A1(b[3]), .A2(a[6]), .B1(b[2]), .B2(a[7]), .ZN(n270) );
  VHSR_IN_2 U268 ( .I(b[3]), .ZN(n323) );
  VHSR_CLKNAND2_2 U269 ( .A1(b[2]), .A2(a[4]), .ZN(n288) );
  VHSR_NOR3_2 U270 ( .A1(n323), .A2(n288), .A3(n238), .ZN(n268) );
  VHSR_IN_2 U271 ( .I(b[1]), .ZN(n392) );
  VHSR_NOR2_1 U272 ( .A1(n256), .A2(n392), .ZN(n233) );
  VHSR_AOI211_2 U273 ( .A1(a[4]), .A2(b[2]), .B(n323), .C(n238), .ZN(n234) );
  VHSR_CLKNAND2_2 U274 ( .A1(a[6]), .A2(b[2]), .ZN(n236) );
  VHSR_IN_2 U275 ( .I(n236), .ZN(n232) );
  VHSR_MAOI222_2 U276 ( .A(n233), .B(n234), .C(n232), .ZN(n245) );
  VHSR_AOI21_2 U277 ( .A1(b[1]), .A2(a[7]), .B(n234), .ZN(n237) );
  VHSR_IN_2 U278 ( .I(n245), .ZN(n235) );
  VHSR_AOI21_2 U279 ( .A1(n237), .A2(n236), .B(n235), .ZN(n273) );
  VHSR_CLKNAND2_2 U280 ( .A1(a[4]), .A2(b[0]), .ZN(n384) );
  VHSR_NOR3_2 U281 ( .A1(n238), .A2(n392), .A3(n384), .ZN(n296) );
  VHSR_AOI22_2 U282 ( .A1(b[3]), .A2(a[4]), .B1(b[2]), .B2(a[5]), .ZN(n239) );
  VHSR_NOR2_1 U283 ( .A1(n268), .A2(n239), .ZN(n240) );
  VHSR_MAOI222_2 U284 ( .A(n241), .B(n296), .C(n240), .ZN(n244) );
  VHSR_NAND3_2 U285 ( .A1(b[1]), .A2(a[5]), .A3(n384), .ZN(n287) );
  VHSR_OAI21_2 U286 ( .A1(a[6]), .A2(a[7]), .B(b[0]), .ZN(n286) );
  VHSR_MAOI222_2 U287 ( .A(n288), .B(n287), .C(n286), .ZN(n285) );
  VHSR_OR2_2 U288 ( .A1(n296), .A2(n240), .Z(n242) );
  VHSR_OAI21_2 U289 ( .A1(n242), .A2(n241), .B(n244), .ZN(n243) );
  VHSR_IN_2 U290 ( .I(n243), .ZN(n281) );
  VHSR_CLKNAND2_2 U291 ( .A1(n285), .A2(n281), .ZN(n280) );
  VHSR_CLKNAND2_2 U292 ( .A1(n244), .A2(n280), .ZN(n272) );
  VHSR_CLKNAND2_2 U293 ( .A1(n273), .A2(n272), .ZN(n271) );
  VHSR_CLKNAND2_2 U294 ( .A1(n245), .A2(n271), .ZN(n267) );
  VHSR_AND3_2 U295 ( .A1(n258), .A2(b[3]), .A3(a[7]), .Z(n306) );
  VHSR_IN_2 U296 ( .I(a[3]), .ZN(n324) );
  VHSR_IN_2 U297 ( .I(a[2]), .ZN(n325) );
  VHSR_IN_2 U298 ( .I(b[6]), .ZN(n249) );
  VHSR_OAI22_2 U299 ( .A1(n325), .A2(n260), .B1(n324), .B2(n249), .ZN(n265) );
  VHSR_NOR2_1 U300 ( .A1(n325), .A2(n260), .ZN(n247) );
  VHSR_IN_2 U301 ( .I(a[1]), .ZN(n389) );
  VHSR_NOR2_1 U302 ( .A1(n249), .A2(n389), .ZN(n246) );
  VHSR_CLKNAND2_2 U303 ( .A1(a[2]), .A2(b[4]), .ZN(n292) );
  VHSR_NAND3_2 U304 ( .A1(a[3]), .A2(b[5]), .A3(n292), .ZN(n253) );
  VHSR_AOI22_2 U305 ( .A1(a[2]), .A2(b[6]), .B1(b[7]), .B2(a[1]), .ZN(n254) );
  VHSR_IAO22_2 U306 ( .B1(n247), .B2(n246), .A1(n253), .A2(n254), .ZN(n255) );
  VHSR_CLKNAND2_2 U307 ( .A1(b[6]), .A2(a[0]), .ZN(n291) );
  VHSR_CLKNAND2_2 U308 ( .A1(b[4]), .A2(a[0]), .ZN(n383) );
  VHSR_NAND3_2 U309 ( .A1(a[1]), .A2(b[5]), .A3(n383), .ZN(n290) );
  VHSR_MAOI222_2 U310 ( .A(n292), .B(n291), .C(n290), .ZN(n289) );
  VHSR_NAND4_2 U311 ( .A1(a[2]), .A2(a[3]), .A3(b[4]), .A4(b[5]), .ZN(n262) );
  VHSR_AND2_2 U312 ( .A1(n262), .A2(n248), .Z(n252) );
  VHSR_OAI22_2 U313 ( .A1(n260), .A2(n391), .B1(n249), .B2(n389), .ZN(n251) );
  VHSR_NOR3_2 U314 ( .A1(n250), .A2(n389), .A3(n383), .ZN(n294) );
  VHSR_AND2_2 U315 ( .A1(n289), .A2(n284), .Z(n283) );
  VHSR_AD1_1 U316 ( .A(n252), .B(n251), .CI(n294), .CO(n275), .S(n284) );
  VHSR_NOR2_1 U317 ( .A1(n283), .A2(n275), .ZN(n278) );
  VHSR_CLKNAND2_2 U318 ( .A1(n263), .A2(n262), .ZN(n261) );
  VHSR_CLKNAND2_2 U319 ( .A1(n265), .A2(n261), .ZN(n259) );
  VHSR_NOR3_2 U320 ( .A1(n260), .A2(n324), .A3(n259), .ZN(n305) );
  VHSR_NOR2_1 U321 ( .A1(n323), .A2(n256), .ZN(n257) );
  VHSR_IAO21_2 U322 ( .A1(n258), .A2(n257), .B(n306), .ZN(n309) );
  VHSR_OAI32_2 U323 ( .A1(n305), .A2(n324), .A3(n260), .B1(n259), .B2(n305), 
        .ZN(n308) );
  VHSR_OAI21_2 U324 ( .A1(n263), .A2(n262), .B(n261), .ZN(n264) );
  VHSR_XNOR2_2 U325 ( .A1(n265), .A2(n264), .ZN(n316) );
  VHSR_AOI21_2 U326 ( .A1(n268), .A2(n267), .B(n266), .ZN(n269) );
  VHSR_XNOR2_2 U327 ( .A1(n270), .A2(n269), .ZN(n315) );
  VHSR_OAI21_2 U328 ( .A1(n273), .A2(n272), .B(n271), .ZN(n274) );
  VHSR_IN_2 U329 ( .I(n274), .ZN(n321) );
  VHSR_CLKNAND2_2 U330 ( .A1(n283), .A2(n275), .ZN(n277) );
  VHSR_AOI22_2 U331 ( .A1(n279), .A2(n278), .B1(n277), .B2(n276), .ZN(n320) );
  VHSR_OAI21_2 U332 ( .A1(n285), .A2(n281), .B(n280), .ZN(n282) );
  VHSR_IN_2 U333 ( .I(n282), .ZN(n330) );
  VHSR_IAO21_2 U334 ( .A1(n289), .A2(n284), .B(n283), .ZN(n329) );
  VHSR_AOI31_2 U335 ( .A1(n288), .A2(n287), .A3(n286), .B(n285), .ZN(n333) );
  VHSR_AOI31_2 U336 ( .A1(n292), .A2(n291), .A3(n290), .B(n289), .ZN(n332) );
  VHSR_AOI22_2 U337 ( .A1(b[4]), .A2(a[1]), .B1(b[5]), .B2(a[0]), .ZN(n293) );
  VHSR_NOR2_1 U338 ( .A1(n294), .A2(n293), .ZN(n346) );
  VHSR_AOI22_2 U339 ( .A1(a[4]), .A2(b[1]), .B1(a[5]), .B2(b[0]), .ZN(n295) );
  VHSR_NOR2_1 U340 ( .A1(n296), .A2(n295), .ZN(n345) );
  VHSR_OAI21_2 U341 ( .A1(n299), .A2(n298), .B(n297), .ZN(n300) );
  VHSR_IN_2 U342 ( .I(n300), .ZN(n349) );
  VHSR_OAI21_2 U343 ( .A1(n310), .A2(n302), .B(n301), .ZN(n303) );
  VHSR_IN_2 U344 ( .I(n303), .ZN(n353) );
  VHSR_AD1_1 U345 ( .A(n309), .B(n308), .CI(n307), .CO(n304), .S(n372) );
  VHSR_AOI31_2 U346 ( .A1(n313), .A2(n312), .A3(n311), .B(n310), .ZN(n371) );
  VHSR_AD1_1 U347 ( .A(n316), .B(n315), .CI(n314), .CO(n307), .S(n356) );
  VHSR_AOI22_2 U348 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n317) );
  VHSR_AD1_1 U349 ( .A(n321), .B(n320), .CI(n319), .CO(n314), .S(n358) );
  VHSR_CLKNAND2_2 U350 ( .A1(b[2]), .A2(a[2]), .ZN(n334) );
  VHSR_CLKNAND2_2 U351 ( .A1(b[3]), .A2(a[3]), .ZN(n337) );
  VHSR_AOI22_2 U352 ( .A1(b[3]), .A2(a[2]), .B1(b[2]), .B2(a[3]), .ZN(n322) );
  VHSR_IAO21_2 U353 ( .A1(n334), .A2(n337), .B(n322), .ZN(n344) );
  VHSR_AOI22_2 U354 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n326) );
  VHSR_CLKNAND2_2 U355 ( .A1(b[1]), .A2(a[1]), .ZN(n396) );
  VHSR_OAI22_2 U356 ( .A1(n334), .A2(n326), .B1(n337), .B2(n396), .ZN(n327) );
  VHSR_CLKNAND2_2 U357 ( .A1(b[0]), .A2(a[2]), .ZN(n395) );
  VHSR_CLKNAND2_2 U358 ( .A1(b[2]), .A2(a[0]), .ZN(n394) );
  VHSR_MAOI222_2 U359 ( .A(n396), .B(n395), .C(n394), .ZN(n393) );
  VHSR_OAI22_2 U360 ( .A1(n392), .A2(n325), .B1(n390), .B2(n324), .ZN(n365) );
  VHSR_AOI21_2 U361 ( .A1(n326), .A2(n334), .B(n327), .ZN(n340) );
  VHSR_CLKNAND2_2 U362 ( .A1(n341), .A2(n340), .ZN(n339) );
  VHSR_CLKNAND2_2 U363 ( .A1(n344), .A2(n343), .ZN(n335) );
  VHSR_AOI21_2 U364 ( .A1(n334), .A2(n335), .B(n337), .ZN(n362) );
  VHSR_AD1_1 U365 ( .A(n330), .B(n329), .CI(n328), .CO(n319), .S(n361) );
  VHSR_AD1_1 U366 ( .A(n333), .B(n332), .CI(n331), .CO(n328), .S(n369) );
  VHSR_IN_2 U367 ( .I(n335), .ZN(n342) );
  VHSR_CLKNAND2_2 U368 ( .A1(n342), .A2(n337), .ZN(n336) );
  VHSR_OAI31_2 U369 ( .A1(n338), .A2(n342), .A3(n337), .B(n336), .ZN(n368) );
  VHSR_OAI21_2 U370 ( .A1(n341), .A2(n340), .B(n339), .ZN(n388) );
  VHSR_AOI211_2 U371 ( .A1(n384), .A2(n383), .B(n382), .C(n388), .ZN(n386) );
  VHSR_IAO21_2 U372 ( .A1(n344), .A2(n343), .B(n342), .ZN(n364) );
  VHSR_AD1_1 U373 ( .A(n346), .B(n345), .CI(n382), .CO(n331), .S(n363) );
  VHSR_AND2_2 U374 ( .A1(n374), .A2(n373), .Z(n381) );
  VHSR_OAI31_2 U375 ( .A1(n378), .A2(n381), .A3(n379), .B(n375), .ZN(n347) );
  VHSR_AD1_1 U376 ( .A(n369), .B(n368), .CI(n367), .CO(n360), .S(product[6])
         );
  VHSR_AD1_1 U377 ( .A(n372), .B(n371), .CI(n370), .CO(n351), .S(product[10])
         );
  VHSR_IAO21_2 U378 ( .A1(n374), .A2(n373), .B(n381), .ZN(product[13]) );
  VHSR_OAI21_2 U379 ( .A1(n379), .A2(n377), .B(n378), .ZN(n376) );
  VHSR_OAI31_2 U380 ( .A1(n379), .A2(n378), .A3(n377), .B(n376), .ZN(n380) );
  VHSR_CLKXOR2_2 U381 ( .A1(n381), .A2(n380), .Z(product[14]) );
  VHSR_AOI21_2 U382 ( .A1(n384), .A2(n383), .B(n382), .ZN(n385) );
  VHSR_IN_2 U383 ( .I(n385), .ZN(n387) );
  VHSR_AOI21_2 U384 ( .A1(n388), .A2(n387), .B(n386), .ZN(product[4]) );
  VHSR_OAI22_2 U385 ( .A1(n392), .A2(n391), .B1(n390), .B2(n389), .ZN(
        product[1]) );
  VHSR_AOI31_2 U386 ( .A1(n396), .A2(n395), .A3(n394), .B(n393), .ZN(
        product[2]) );
endmodule

