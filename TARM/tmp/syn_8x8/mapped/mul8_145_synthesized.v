
module mul8_145 ( a, b, product );
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
         n382, n383, n384, n385, n386, n387, n388, n389, n390, n391;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U195 ( .A1(n230), .B1(n249), .ZN(n236) );
  VHSR_NOR2_1 U196 ( .A1(n335), .A2(n334), .ZN(n333) );
  VHSR_INOR3_2 U197 ( .A1(n232), .B1(n318), .B2(n275), .ZN(n293) );
  VHSR_NOR2_1 U198 ( .A1(n387), .A2(n386), .ZN(n385) );
  VHSR_INOR2_2 U199 ( .A1(n354), .B1(n353), .ZN(n383) );
  VHSR_IN_2 U200 ( .I(n350), .ZN(product[13]) );
  VHSR_INAND2_1 U201 ( .A1(n345), .B1(n343), .ZN(n346) );
  VHSR_NOR2_2 U202 ( .A1(n315), .A2(n316), .ZN(n326) );
  VHSR_AD1_1 U203 ( .A(n362), .B(n361), .CI(n360), .CO(n357), .S(product[9])
         );
  VHSR_AD1_1 U204 ( .A(n385), .B(n370), .CI(n369), .CO(n371), .S(product[5])
         );
  VHSR_AD1_1 U205 ( .A(n368), .B(n367), .CI(n366), .CO(n363), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U206 ( .A(n365), .B(n364), .CI(n363), .CO(n360), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U207 ( .A(n359), .B(n358), .CI(n357), .CO(n374), .S(product[10])
         );
  VHSR_IN_2 U208 ( .I(b[2]), .ZN(n315) );
  VHSR_IN_2 U209 ( .I(a[2]), .ZN(n316) );
  VHSR_IN_2 U210 ( .I(b[0]), .ZN(n389) );
  VHSR_IN_2 U211 ( .I(a[0]), .ZN(n390) );
  VHSR_NOR2_1 U212 ( .A1(n389), .A2(n390), .ZN(product[0]) );
  VHSR_IN_2 U213 ( .I(b[1]), .ZN(n391) );
  VHSR_NOR2_1 U214 ( .A1(n391), .A2(n390), .ZN(n204) );
  VHSR_IN_2 U215 ( .I(a[1]), .ZN(n388) );
  VHSR_NOR2_1 U216 ( .A1(n315), .A2(n388), .ZN(n206) );
  VHSR_AOI22_2 U217 ( .A1(n326), .A2(product[0]), .B1(n204), .B2(n206), .ZN(
        n314) );
  VHSR_OAI222_2 U218 ( .A1(n316), .A2(n389), .B1(n390), .B2(n315), .C1(n388), 
        .C2(n391), .ZN(n205) );
  VHSR_AND2_2 U219 ( .A1(n314), .A2(n205), .Z(product[2]) );
  VHSR_AOI22_2 U220 ( .A1(b[1]), .A2(a[2]), .B1(b[0]), .B2(a[3]), .ZN(n313) );
  VHSR_AOI21_2 U221 ( .A1(a[0]), .A2(b[3]), .B(n206), .ZN(n312) );
  VHSR_IN_2 U222 ( .I(n207), .ZN(product[3]) );
  VHSR_AOI22_2 U223 ( .A1(a[6]), .A2(b[3]), .B1(a[7]), .B2(b[2]), .ZN(n243) );
  VHSR_IN_2 U224 ( .I(b[3]), .ZN(n318) );
  VHSR_IN_2 U225 ( .I(a[5]), .ZN(n280) );
  VHSR_IN_2 U226 ( .I(a[4]), .ZN(n279) );
  VHSR_NOR4_2 U227 ( .A1(n318), .A2(n315), .A3(n280), .A4(n279), .ZN(n241) );
  VHSR_IN_2 U228 ( .I(a[7]), .ZN(n275) );
  VHSR_NOR2_1 U229 ( .A1(n275), .A2(n391), .ZN(n209) );
  VHSR_AOI211_2 U230 ( .A1(b[2]), .A2(a[4]), .B(n318), .C(n280), .ZN(n210) );
  VHSR_CLKNAND2_2 U231 ( .A1(a[6]), .A2(b[2]), .ZN(n212) );
  VHSR_IN_2 U232 ( .I(n212), .ZN(n208) );
  VHSR_MAOI222_2 U233 ( .A(n209), .B(n210), .C(n208), .ZN(n222) );
  VHSR_AOI21_2 U234 ( .A1(b[1]), .A2(a[7]), .B(n210), .ZN(n213) );
  VHSR_IN_2 U235 ( .I(n222), .ZN(n211) );
  VHSR_AOI21_2 U236 ( .A1(n213), .A2(n212), .B(n211), .ZN(n246) );
  VHSR_NOR4_2 U237 ( .A1(n280), .A2(n279), .A3(n391), .A4(n389), .ZN(n269) );
  VHSR_CLKNAND2_2 U238 ( .A1(b[2]), .A2(a[5]), .ZN(n215) );
  VHSR_CLKNAND2_2 U239 ( .A1(b[3]), .A2(a[4]), .ZN(n214) );
  VHSR_AOI21_2 U240 ( .A1(n215), .A2(n214), .B(n241), .ZN(n217) );
  VHSR_AOI22_2 U241 ( .A1(a[6]), .A2(b[1]), .B1(a[7]), .B2(b[0]), .ZN(n219) );
  VHSR_IN_2 U242 ( .I(n219), .ZN(n216) );
  VHSR_MAOI222_2 U243 ( .A(n269), .B(n217), .C(n216), .ZN(n221) );
  VHSR_CLKNAND2_2 U244 ( .A1(b[2]), .A2(a[4]), .ZN(n265) );
  VHSR_OAI211_2 U245 ( .A1(n279), .A2(n389), .B(a[5]), .C(b[1]), .ZN(n264) );
  VHSR_CLKNAND2_2 U246 ( .A1(a[6]), .A2(b[0]), .ZN(n263) );
  VHSR_MAOI222_2 U247 ( .A(n265), .B(n264), .C(n263), .ZN(n262) );
  VHSR_NOR2_1 U248 ( .A1(n269), .A2(n217), .ZN(n220) );
  VHSR_IN_2 U249 ( .I(n221), .ZN(n218) );
  VHSR_AOI21_2 U250 ( .A1(n220), .A2(n219), .B(n218), .ZN(n256) );
  VHSR_CLKNAND2_2 U251 ( .A1(n262), .A2(n256), .ZN(n255) );
  VHSR_CLKNAND2_2 U252 ( .A1(n221), .A2(n255), .ZN(n245) );
  VHSR_CLKNAND2_2 U253 ( .A1(n246), .A2(n245), .ZN(n244) );
  VHSR_CLKNAND2_2 U254 ( .A1(n222), .A2(n244), .ZN(n240) );
  VHSR_NOR2_1 U255 ( .A1(n241), .A2(n240), .ZN(n239) );
  VHSR_NOR2_1 U256 ( .A1(n243), .A2(n239), .ZN(n232) );
  VHSR_IN_2 U257 ( .I(b[7]), .ZN(n277) );
  VHSR_IN_2 U258 ( .I(a[3]), .ZN(n319) );
  VHSR_IN_2 U259 ( .I(b[6]), .ZN(n278) );
  VHSR_OAI22_2 U260 ( .A1(n278), .A2(n319), .B1(n277), .B2(n316), .ZN(n238) );
  VHSR_NOR2_1 U261 ( .A1(n277), .A2(n316), .ZN(n224) );
  VHSR_NOR2_1 U262 ( .A1(n278), .A2(n388), .ZN(n223) );
  VHSR_IN_2 U263 ( .I(b[5]), .ZN(n274) );
  VHSR_AOI211_2 U264 ( .A1(b[4]), .A2(a[2]), .B(n274), .C(n319), .ZN(n229) );
  VHSR_OAI22_2 U265 ( .A1(n278), .A2(n316), .B1(n277), .B2(n388), .ZN(n228) );
  VHSR_AOI22_2 U266 ( .A1(n224), .A2(n223), .B1(n229), .B2(n228), .ZN(n230) );
  VHSR_CLKNAND2_2 U267 ( .A1(b[4]), .A2(a[2]), .ZN(n261) );
  VHSR_IN_2 U268 ( .I(b[4]), .ZN(n330) );
  VHSR_OAI211_2 U269 ( .A1(n330), .A2(n390), .B(b[5]), .C(a[1]), .ZN(n260) );
  VHSR_CLKNAND2_2 U270 ( .A1(b[6]), .A2(a[0]), .ZN(n259) );
  VHSR_MAOI222_2 U271 ( .A(n261), .B(n260), .C(n259), .ZN(n258) );
  VHSR_NOR4_2 U272 ( .A1(n330), .A2(n274), .A3(n388), .A4(n390), .ZN(n267) );
  VHSR_NAND4_2 U273 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n235) );
  VHSR_OAI22_2 U274 ( .A1(n330), .A2(n319), .B1(n274), .B2(n316), .ZN(n225) );
  VHSR_AND2_2 U275 ( .A1(n235), .A2(n225), .Z(n227) );
  VHSR_OAI22_2 U276 ( .A1(n278), .A2(n388), .B1(n277), .B2(n390), .ZN(n226) );
  VHSR_AND2_2 U277 ( .A1(n258), .A2(n254), .Z(n253) );
  VHSR_AD1_1 U278 ( .A(n267), .B(n227), .CI(n226), .CO(n248), .S(n254) );
  VHSR_NOR2_1 U279 ( .A1(n253), .A2(n248), .ZN(n251) );
  VHSR_OAI21_2 U280 ( .A1(n229), .A2(n228), .B(n230), .ZN(n252) );
  VHSR_NOR2_1 U281 ( .A1(n251), .A2(n252), .ZN(n249) );
  VHSR_CLKNAND2_2 U282 ( .A1(n236), .A2(n235), .ZN(n234) );
  VHSR_CLKNAND2_2 U283 ( .A1(n238), .A2(n234), .ZN(n233) );
  VHSR_NOR3_2 U284 ( .A1(n277), .A2(n319), .A3(n233), .ZN(n292) );
  VHSR_NOR2_1 U285 ( .A1(n318), .A2(n275), .ZN(n231) );
  VHSR_IAO21_2 U286 ( .A1(n232), .A2(n231), .B(n293), .ZN(n296) );
  VHSR_OAI32_2 U287 ( .A1(n292), .A2(n319), .A3(n277), .B1(n233), .B2(n292), 
        .ZN(n295) );
  VHSR_OAI21_2 U288 ( .A1(n236), .A2(n235), .B(n234), .ZN(n237) );
  VHSR_XNOR2_2 U289 ( .A1(n238), .A2(n237), .ZN(n303) );
  VHSR_AOI21_2 U290 ( .A1(n241), .A2(n240), .B(n239), .ZN(n242) );
  VHSR_XNOR2_2 U291 ( .A1(n243), .A2(n242), .ZN(n302) );
  VHSR_OAI21_2 U292 ( .A1(n246), .A2(n245), .B(n244), .ZN(n247) );
  VHSR_IN_2 U293 ( .I(n247), .ZN(n308) );
  VHSR_CLKNAND2_2 U294 ( .A1(n253), .A2(n248), .ZN(n250) );
  VHSR_AOI22_2 U295 ( .A1(n252), .A2(n251), .B1(n250), .B2(n249), .ZN(n307) );
  VHSR_IAO21_2 U296 ( .A1(n258), .A2(n254), .B(n253), .ZN(n311) );
  VHSR_OAI21_2 U297 ( .A1(n262), .A2(n256), .B(n255), .ZN(n257) );
  VHSR_IN_2 U298 ( .I(n257), .ZN(n310) );
  VHSR_AOI31_2 U299 ( .A1(n261), .A2(n260), .A3(n259), .B(n258), .ZN(n323) );
  VHSR_AOI31_2 U300 ( .A1(n265), .A2(n264), .A3(n263), .B(n262), .ZN(n322) );
  VHSR_CLKNAND2_2 U301 ( .A1(b[5]), .A2(a[0]), .ZN(n266) );
  VHSR_OAI32_2 U302 ( .A1(n267), .A2(n388), .A3(n330), .B1(n266), .B2(n267), 
        .ZN(n338) );
  VHSR_CLKNAND2_2 U303 ( .A1(a[4]), .A2(b[1]), .ZN(n268) );
  VHSR_OAI32_2 U304 ( .A1(n269), .A2(n389), .A3(n280), .B1(n268), .B2(n269), 
        .ZN(n337) );
  VHSR_CLKNAND2_2 U305 ( .A1(a[4]), .A2(b[4]), .ZN(n282) );
  VHSR_IN_2 U306 ( .I(n282), .ZN(n364) );
  VHSR_CLKNAND2_2 U307 ( .A1(n364), .A2(product[0]), .ZN(n332) );
  VHSR_IN_2 U308 ( .I(n332), .ZN(n336) );
  VHSR_CLKNAND2_2 U309 ( .A1(a[6]), .A2(b[6]), .ZN(n355) );
  VHSR_IN_2 U310 ( .I(n355), .ZN(n380) );
  VHSR_CLKNAND2_2 U311 ( .A1(a[6]), .A2(b[4]), .ZN(n300) );
  VHSR_NAND3_2 U312 ( .A1(a[7]), .A2(b[5]), .A3(n300), .ZN(n271) );
  VHSR_CLKNAND2_2 U313 ( .A1(a[4]), .A2(b[6]), .ZN(n299) );
  VHSR_NAND3_2 U314 ( .A1(b[7]), .A2(a[5]), .A3(n299), .ZN(n270) );
  VHSR_CLKNAND2_2 U315 ( .A1(n271), .A2(n270), .ZN(n273) );
  VHSR_MAOI222_2 U316 ( .A(n355), .B(n271), .C(n270), .ZN(n339) );
  VHSR_IN_2 U317 ( .I(n339), .ZN(n272) );
  VHSR_OAI21_2 U318 ( .A1(n380), .A2(n273), .B(n272), .ZN(n288) );
  VHSR_NOR3_2 U319 ( .A1(n280), .A2(n274), .A3(n282), .ZN(n304) );
  VHSR_NOR3_2 U320 ( .A1(n275), .A2(n300), .A3(n274), .ZN(n347) );
  VHSR_AOI22_2 U321 ( .A1(a[6]), .A2(b[5]), .B1(a[7]), .B2(b[4]), .ZN(n276) );
  VHSR_NOR2_1 U322 ( .A1(n347), .A2(n276), .ZN(n284) );
  VHSR_NOR4_2 U323 ( .A1(n280), .A2(n279), .A3(n278), .A4(n277), .ZN(n345) );
  VHSR_AOI22_2 U324 ( .A1(a[5]), .A2(b[6]), .B1(a[4]), .B2(b[7]), .ZN(n281) );
  VHSR_NOR2_1 U325 ( .A1(n345), .A2(n281), .ZN(n283) );
  VHSR_NAND3_2 U326 ( .A1(b[5]), .A2(a[5]), .A3(n282), .ZN(n298) );
  VHSR_MAOI222_2 U327 ( .A(n300), .B(n299), .C(n298), .ZN(n297) );
  VHSR_AND2_2 U328 ( .A1(n290), .A2(n297), .Z(n289) );
  VHSR_AD1_1 U329 ( .A(n304), .B(n284), .CI(n283), .CO(n285), .S(n290) );
  VHSR_NOR2_1 U330 ( .A1(n289), .A2(n285), .ZN(n287) );
  VHSR_CLKNAND2_2 U331 ( .A1(n289), .A2(n285), .ZN(n286) );
  VHSR_NOR2_1 U332 ( .A1(n287), .A2(n288), .ZN(n340) );
  VHSR_AOI22_2 U333 ( .A1(n288), .A2(n287), .B1(n286), .B2(n340), .ZN(n378) );
  VHSR_IAO21_2 U334 ( .A1(n290), .A2(n297), .B(n289), .ZN(n376) );
  VHSR_AD1_1 U335 ( .A(n293), .B(n292), .CI(n291), .CO(n379), .S(n375) );
  VHSR_AD1_1 U336 ( .A(n296), .B(n295), .CI(n294), .CO(n291), .S(n359) );
  VHSR_AOI31_2 U337 ( .A1(n300), .A2(n299), .A3(n298), .B(n297), .ZN(n358) );
  VHSR_AD1_1 U338 ( .A(n303), .B(n302), .CI(n301), .CO(n294), .S(n362) );
  VHSR_AOI22_2 U339 ( .A1(a[5]), .A2(b[4]), .B1(a[4]), .B2(b[5]), .ZN(n305) );
  VHSR_NOR2_1 U340 ( .A1(n305), .A2(n304), .ZN(n361) );
  VHSR_AD1_1 U341 ( .A(n308), .B(n307), .CI(n306), .CO(n301), .S(n365) );
  VHSR_AD1_1 U342 ( .A(n311), .B(n310), .CI(n309), .CO(n306), .S(n368) );
  VHSR_IN_2 U343 ( .I(n326), .ZN(n329) );
  VHSR_AD1_1 U344 ( .A(n314), .B(n313), .CI(n312), .CO(n328), .S(n207) );
  VHSR_AOI22_2 U345 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n327) );
  VHSR_OAI22_2 U346 ( .A1(n318), .A2(n316), .B1(n315), .B2(n319), .ZN(n317) );
  VHSR_OAI31_2 U347 ( .A1(n319), .A2(n318), .A3(n329), .B(n317), .ZN(n334) );
  VHSR_OAI211_2 U348 ( .A1(n326), .A2(n333), .B(a[3]), .C(b[3]), .ZN(n320) );
  VHSR_IN_2 U349 ( .I(n320), .ZN(n367) );
  VHSR_AD1_1 U350 ( .A(n323), .B(n322), .CI(n321), .CO(n309), .S(n373) );
  VHSR_CLKNAND2_2 U351 ( .A1(b[3]), .A2(a[3]), .ZN(n325) );
  VHSR_CLKNAND2_2 U352 ( .A1(n333), .A2(n325), .ZN(n324) );
  VHSR_OAI31_2 U353 ( .A1(n326), .A2(n333), .A3(n325), .B(n324), .ZN(n372) );
  VHSR_AD1_1 U354 ( .A(n329), .B(n328), .CI(n327), .CO(n335), .S(n387) );
  VHSR_NOR2_1 U355 ( .A1(n330), .A2(n390), .ZN(n331) );
  VHSR_AOI32_2 U356 ( .A1(b[0]), .A2(n332), .A3(a[4]), .B1(n331), .B2(n332), 
        .ZN(n386) );
  VHSR_AOI21_2 U357 ( .A1(n335), .A2(n334), .B(n333), .ZN(n370) );
  VHSR_AD1_1 U358 ( .A(n338), .B(n337), .CI(n336), .CO(n321), .S(n369) );
  VHSR_NOR2_1 U359 ( .A1(n340), .A2(n339), .ZN(n352) );
  VHSR_CLKNAND2_2 U360 ( .A1(a[7]), .A2(b[6]), .ZN(n342) );
  VHSR_AOI21_2 U361 ( .A1(a[6]), .A2(b[7]), .B(n342), .ZN(n341) );
  VHSR_AOI31_2 U362 ( .A1(a[6]), .A2(n342), .A3(b[7]), .B(n341), .ZN(n343) );
  VHSR_IN_2 U363 ( .I(n343), .ZN(n344) );
  VHSR_MAOI222_2 U364 ( .A(n347), .B(n345), .C(n344), .ZN(n354) );
  VHSR_OAI21_2 U365 ( .A1(n347), .A2(n346), .B(n354), .ZN(n351) );
  VHSR_CLKXOR2_2 U366 ( .A1(n352), .A2(n351), .Z(n348) );
  VHSR_CLKNAND2_2 U367 ( .A1(n349), .A2(n348), .ZN(n382) );
  VHSR_OAI21_2 U368 ( .A1(n349), .A2(n348), .B(n382), .ZN(n350) );
  VHSR_CLKNAND2_2 U369 ( .A1(a[7]), .A2(b[7]), .ZN(n381) );
  VHSR_NOR2_1 U370 ( .A1(n352), .A2(n351), .ZN(n353) );
  VHSR_AND3_2 U371 ( .A1(n383), .A2(n355), .A3(n382), .Z(n356) );
  VHSR_NOR2_1 U372 ( .A1(n381), .A2(n356), .ZN(product[15]) );
  VHSR_AD1_1 U373 ( .A(n373), .B(n372), .CI(n371), .CO(n366), .S(product[6])
         );
  VHSR_AD1_1 U374 ( .A(n376), .B(n375), .CI(n374), .CO(n377), .S(product[11])
         );
  VHSR_AD1_1 U375 ( .A(n379), .B(n378), .CI(n377), .CO(n349), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U376 ( .A1(n381), .A2(n380), .ZN(n384) );
  VHSR_XOR3_2 U377 ( .A1(n384), .A2(n383), .A3(n382), .Z(product[14]) );
  VHSR_AOI21_2 U378 ( .A1(n387), .A2(n386), .B(n385), .ZN(product[4]) );
  VHSR_OAI22_2 U379 ( .A1(n391), .A2(n390), .B1(n389), .B2(n388), .ZN(
        product[1]) );
endmodule

