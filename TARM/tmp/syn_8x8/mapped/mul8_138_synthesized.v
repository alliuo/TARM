
module mul8_138 ( a, b, product );
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
         n389, n390, n391;
  assign product[0] = \mul_ll_ll/out[0] ;
  assign product[12] = \intadd_0/SUM[7] ;
  assign product[8] = \intadd_0/SUM[3] ;
  assign product[7] = \intadd_0/SUM[2] ;

  VHSR_INOR2_2 U196 ( .A1(n231), .B1(n250), .ZN(n237) );
  VHSR_NOR2_1 U197 ( .A1(n339), .A2(n338), .ZN(n337) );
  VHSR_INOR3_2 U198 ( .A1(n233), .B1(n278), .B2(n321), .ZN(n296) );
  VHSR_NOR2_1 U199 ( .A1(n391), .A2(n390), .ZN(n389) );
  VHSR_INOR2_2 U200 ( .A1(n358), .B1(n357), .ZN(n387) );
  VHSR_IN_2 U201 ( .I(n354), .ZN(product[13]) );
  VHSR_INAND2_1 U202 ( .A1(n349), .B1(n347), .ZN(n350) );
  VHSR_NOR2_2 U203 ( .A1(n319), .A2(n318), .ZN(n329) );
  VHSR_AD1_1 U204 ( .A(n366), .B(n365), .CI(n364), .CO(n361), .S(product[9])
         );
  VHSR_AD1_1 U205 ( .A(n389), .B(n374), .CI(n373), .CO(n375), .S(product[5])
         );
  VHSR_AD1_1 U206 ( .A(n372), .B(n371), .CI(n370), .CO(n367), .S(
        \intadd_0/SUM[2] ) );
  VHSR_AD1_1 U207 ( .A(n369), .B(n368), .CI(n367), .CO(n364), .S(
        \intadd_0/SUM[3] ) );
  VHSR_AD1_1 U208 ( .A(n363), .B(n362), .CI(n361), .CO(n378), .S(product[10])
         );
  VHSR_PULL0_0 U209 ( .Z(\mul_ll_ll/out[0] ) );
  VHSR_IN_2 U210 ( .I(b[2]), .ZN(n319) );
  VHSR_IN_2 U211 ( .I(a[2]), .ZN(n318) );
  VHSR_IN_2 U212 ( .I(b[0]), .ZN(n217) );
  VHSR_IN_2 U213 ( .I(a[0]), .ZN(n333) );
  VHSR_NOR2_1 U214 ( .A1(n217), .A2(n333), .ZN(n205) );
  VHSR_IN_2 U215 ( .I(b[1]), .ZN(n271) );
  VHSR_NOR2_1 U216 ( .A1(n271), .A2(n333), .ZN(product[1]) );
  VHSR_IN_2 U217 ( .I(a[1]), .ZN(n268) );
  VHSR_NOR2_1 U218 ( .A1(n319), .A2(n268), .ZN(n207) );
  VHSR_AOI22_2 U219 ( .A1(n329), .A2(n205), .B1(product[1]), .B2(n207), .ZN(
        n317) );
  VHSR_OAI222_2 U220 ( .A1(n318), .A2(n217), .B1(n333), .B2(n319), .C1(n268), 
        .C2(n271), .ZN(n206) );
  VHSR_AND2_2 U221 ( .A1(n317), .A2(n206), .Z(product[2]) );
  VHSR_AOI22_2 U222 ( .A1(b[1]), .A2(a[2]), .B1(b[0]), .B2(a[3]), .ZN(n316) );
  VHSR_AOI21_2 U223 ( .A1(a[0]), .A2(b[3]), .B(n207), .ZN(n315) );
  VHSR_IN_2 U224 ( .I(n208), .ZN(product[3]) );
  VHSR_AOI22_2 U225 ( .A1(a[7]), .A2(b[2]), .B1(a[6]), .B2(b[3]), .ZN(n244) );
  VHSR_IN_2 U226 ( .I(b[3]), .ZN(n321) );
  VHSR_CLKNAND2_2 U227 ( .A1(b[2]), .A2(a[4]), .ZN(n266) );
  VHSR_IN_2 U228 ( .I(a[5]), .ZN(n282) );
  VHSR_NOR3_2 U229 ( .A1(n321), .A2(n266), .A3(n282), .ZN(n242) );
  VHSR_IN_2 U230 ( .I(a[7]), .ZN(n278) );
  VHSR_NOR2_1 U231 ( .A1(n278), .A2(n271), .ZN(n210) );
  VHSR_AOI211_2 U232 ( .A1(b[2]), .A2(a[4]), .B(n321), .C(n282), .ZN(n211) );
  VHSR_CLKNAND2_2 U233 ( .A1(b[2]), .A2(a[6]), .ZN(n213) );
  VHSR_IN_2 U234 ( .I(n213), .ZN(n209) );
  VHSR_MAOI222_2 U235 ( .A(n210), .B(n211), .C(n209), .ZN(n223) );
  VHSR_AOI21_2 U236 ( .A1(b[1]), .A2(a[7]), .B(n211), .ZN(n214) );
  VHSR_IN_2 U237 ( .I(n223), .ZN(n212) );
  VHSR_AOI21_2 U238 ( .A1(n214), .A2(n213), .B(n212), .ZN(n247) );
  VHSR_IN_2 U239 ( .I(a[4]), .ZN(n283) );
  VHSR_NOR4_2 U240 ( .A1(n283), .A2(n282), .A3(n271), .A4(n217), .ZN(n272) );
  VHSR_AOI22_2 U241 ( .A1(b[2]), .A2(a[5]), .B1(b[3]), .B2(a[4]), .ZN(n215) );
  VHSR_NOR2_1 U242 ( .A1(n242), .A2(n215), .ZN(n218) );
  VHSR_AOI22_2 U243 ( .A1(a[7]), .A2(b[0]), .B1(a[6]), .B2(b[1]), .ZN(n220) );
  VHSR_IN_2 U244 ( .I(n220), .ZN(n216) );
  VHSR_MAOI222_2 U245 ( .A(n272), .B(n218), .C(n216), .ZN(n222) );
  VHSR_OAI211_2 U246 ( .A1(n283), .A2(n217), .B(a[5]), .C(b[1]), .ZN(n265) );
  VHSR_CLKNAND2_2 U247 ( .A1(a[6]), .A2(b[0]), .ZN(n264) );
  VHSR_MAOI222_2 U248 ( .A(n266), .B(n265), .C(n264), .ZN(n263) );
  VHSR_NOR2_1 U249 ( .A1(n272), .A2(n218), .ZN(n221) );
  VHSR_IN_2 U250 ( .I(n222), .ZN(n219) );
  VHSR_AOI21_2 U251 ( .A1(n221), .A2(n220), .B(n219), .ZN(n257) );
  VHSR_CLKNAND2_2 U252 ( .A1(n263), .A2(n257), .ZN(n256) );
  VHSR_CLKNAND2_2 U253 ( .A1(n222), .A2(n256), .ZN(n246) );
  VHSR_CLKNAND2_2 U254 ( .A1(n247), .A2(n246), .ZN(n245) );
  VHSR_CLKNAND2_2 U255 ( .A1(n223), .A2(n245), .ZN(n241) );
  VHSR_NOR2_1 U256 ( .A1(n242), .A2(n241), .ZN(n240) );
  VHSR_NOR2_1 U257 ( .A1(n244), .A2(n240), .ZN(n233) );
  VHSR_IN_2 U258 ( .I(b[7]), .ZN(n280) );
  VHSR_IN_2 U259 ( .I(a[3]), .ZN(n322) );
  VHSR_IN_2 U260 ( .I(b[6]), .ZN(n281) );
  VHSR_OAI22_2 U261 ( .A1(n281), .A2(n322), .B1(n280), .B2(n318), .ZN(n239) );
  VHSR_NOR2_1 U262 ( .A1(n280), .A2(n318), .ZN(n225) );
  VHSR_NOR2_1 U263 ( .A1(n281), .A2(n268), .ZN(n224) );
  VHSR_IN_2 U264 ( .I(b[5]), .ZN(n277) );
  VHSR_AOI211_2 U265 ( .A1(b[4]), .A2(a[2]), .B(n277), .C(n322), .ZN(n230) );
  VHSR_OAI22_2 U266 ( .A1(n281), .A2(n318), .B1(n280), .B2(n268), .ZN(n229) );
  VHSR_AOI22_2 U267 ( .A1(n225), .A2(n224), .B1(n230), .B2(n229), .ZN(n231) );
  VHSR_CLKNAND2_2 U268 ( .A1(b[4]), .A2(a[2]), .ZN(n262) );
  VHSR_IN_2 U269 ( .I(b[4]), .ZN(n334) );
  VHSR_OAI211_2 U270 ( .A1(n334), .A2(n333), .B(b[5]), .C(a[1]), .ZN(n261) );
  VHSR_CLKNAND2_2 U271 ( .A1(b[6]), .A2(a[0]), .ZN(n260) );
  VHSR_MAOI222_2 U272 ( .A(n262), .B(n261), .C(n260), .ZN(n259) );
  VHSR_NOR4_2 U273 ( .A1(n334), .A2(n277), .A3(n268), .A4(n333), .ZN(n269) );
  VHSR_NAND4_2 U274 ( .A1(b[4]), .A2(b[5]), .A3(a[2]), .A4(a[3]), .ZN(n236) );
  VHSR_OAI22_2 U275 ( .A1(n334), .A2(n322), .B1(n277), .B2(n318), .ZN(n226) );
  VHSR_AND2_2 U276 ( .A1(n236), .A2(n226), .Z(n228) );
  VHSR_OAI22_2 U277 ( .A1(n281), .A2(n268), .B1(n280), .B2(n333), .ZN(n227) );
  VHSR_AND2_2 U278 ( .A1(n259), .A2(n255), .Z(n254) );
  VHSR_AD1_1 U279 ( .A(n269), .B(n228), .CI(n227), .CO(n249), .S(n255) );
  VHSR_NOR2_1 U280 ( .A1(n254), .A2(n249), .ZN(n252) );
  VHSR_OAI21_2 U281 ( .A1(n230), .A2(n229), .B(n231), .ZN(n253) );
  VHSR_NOR2_1 U282 ( .A1(n252), .A2(n253), .ZN(n250) );
  VHSR_CLKNAND2_2 U283 ( .A1(n237), .A2(n236), .ZN(n235) );
  VHSR_CLKNAND2_2 U284 ( .A1(n239), .A2(n235), .ZN(n234) );
  VHSR_NOR3_2 U285 ( .A1(n280), .A2(n322), .A3(n234), .ZN(n295) );
  VHSR_NOR2_1 U286 ( .A1(n278), .A2(n321), .ZN(n232) );
  VHSR_IAO21_2 U287 ( .A1(n233), .A2(n232), .B(n296), .ZN(n299) );
  VHSR_OAI32_2 U288 ( .A1(n295), .A2(n322), .A3(n280), .B1(n234), .B2(n295), 
        .ZN(n298) );
  VHSR_OAI21_2 U289 ( .A1(n237), .A2(n236), .B(n235), .ZN(n238) );
  VHSR_XNOR2_2 U290 ( .A1(n239), .A2(n238), .ZN(n306) );
  VHSR_AOI21_2 U291 ( .A1(n242), .A2(n241), .B(n240), .ZN(n243) );
  VHSR_XNOR2_2 U292 ( .A1(n244), .A2(n243), .ZN(n305) );
  VHSR_OAI21_2 U293 ( .A1(n247), .A2(n246), .B(n245), .ZN(n248) );
  VHSR_IN_2 U294 ( .I(n248), .ZN(n311) );
  VHSR_CLKNAND2_2 U295 ( .A1(n254), .A2(n249), .ZN(n251) );
  VHSR_AOI22_2 U296 ( .A1(n253), .A2(n252), .B1(n251), .B2(n250), .ZN(n310) );
  VHSR_IAO21_2 U297 ( .A1(n259), .A2(n255), .B(n254), .ZN(n314) );
  VHSR_OAI21_2 U298 ( .A1(n263), .A2(n257), .B(n256), .ZN(n258) );
  VHSR_IN_2 U299 ( .I(n258), .ZN(n313) );
  VHSR_AOI31_2 U300 ( .A1(n262), .A2(n261), .A3(n260), .B(n259), .ZN(n326) );
  VHSR_AOI31_2 U301 ( .A1(n266), .A2(n265), .A3(n264), .B(n263), .ZN(n325) );
  VHSR_CLKNAND2_2 U302 ( .A1(b[5]), .A2(a[0]), .ZN(n267) );
  VHSR_OAI32_2 U303 ( .A1(n269), .A2(n268), .A3(n334), .B1(n267), .B2(n269), 
        .ZN(n342) );
  VHSR_CLKNAND2_2 U304 ( .A1(a[5]), .A2(b[0]), .ZN(n270) );
  VHSR_OAI32_2 U305 ( .A1(n272), .A2(n271), .A3(n283), .B1(n270), .B2(n272), 
        .ZN(n341) );
  VHSR_CLKNAND2_2 U306 ( .A1(a[4]), .A2(b[4]), .ZN(n285) );
  VHSR_IN_2 U307 ( .I(n285), .ZN(n368) );
  VHSR_NAND3_2 U308 ( .A1(b[0]), .A2(n368), .A3(a[0]), .ZN(n336) );
  VHSR_IN_2 U309 ( .I(n336), .ZN(n340) );
  VHSR_CLKNAND2_2 U310 ( .A1(a[6]), .A2(b[6]), .ZN(n359) );
  VHSR_IN_2 U311 ( .I(n359), .ZN(n384) );
  VHSR_CLKNAND2_2 U312 ( .A1(a[6]), .A2(b[4]), .ZN(n303) );
  VHSR_NAND3_2 U313 ( .A1(a[7]), .A2(b[5]), .A3(n303), .ZN(n274) );
  VHSR_CLKNAND2_2 U314 ( .A1(a[4]), .A2(b[6]), .ZN(n302) );
  VHSR_NAND3_2 U315 ( .A1(b[7]), .A2(a[5]), .A3(n302), .ZN(n273) );
  VHSR_CLKNAND2_2 U316 ( .A1(n274), .A2(n273), .ZN(n276) );
  VHSR_MAOI222_2 U317 ( .A(n359), .B(n274), .C(n273), .ZN(n343) );
  VHSR_IN_2 U318 ( .I(n343), .ZN(n275) );
  VHSR_OAI21_2 U319 ( .A1(n384), .A2(n276), .B(n275), .ZN(n291) );
  VHSR_NOR3_2 U320 ( .A1(n282), .A2(n277), .A3(n285), .ZN(n307) );
  VHSR_NOR3_2 U321 ( .A1(n278), .A2(n303), .A3(n277), .ZN(n351) );
  VHSR_AOI22_2 U322 ( .A1(a[7]), .A2(b[4]), .B1(a[6]), .B2(b[5]), .ZN(n279) );
  VHSR_NOR2_1 U323 ( .A1(n351), .A2(n279), .ZN(n287) );
  VHSR_NOR4_2 U324 ( .A1(n283), .A2(n282), .A3(n281), .A4(n280), .ZN(n349) );
  VHSR_AOI22_2 U325 ( .A1(a[4]), .A2(b[7]), .B1(a[5]), .B2(b[6]), .ZN(n284) );
  VHSR_NOR2_1 U326 ( .A1(n349), .A2(n284), .ZN(n286) );
  VHSR_NAND3_2 U327 ( .A1(b[5]), .A2(a[5]), .A3(n285), .ZN(n301) );
  VHSR_MAOI222_2 U328 ( .A(n303), .B(n302), .C(n301), .ZN(n300) );
  VHSR_AND2_2 U329 ( .A1(n293), .A2(n300), .Z(n292) );
  VHSR_AD1_1 U330 ( .A(n307), .B(n287), .CI(n286), .CO(n288), .S(n293) );
  VHSR_NOR2_1 U331 ( .A1(n292), .A2(n288), .ZN(n290) );
  VHSR_CLKNAND2_2 U332 ( .A1(n292), .A2(n288), .ZN(n289) );
  VHSR_NOR2_1 U333 ( .A1(n290), .A2(n291), .ZN(n344) );
  VHSR_AOI22_2 U334 ( .A1(n291), .A2(n290), .B1(n289), .B2(n344), .ZN(n382) );
  VHSR_IAO21_2 U335 ( .A1(n293), .A2(n300), .B(n292), .ZN(n380) );
  VHSR_AD1_1 U336 ( .A(n296), .B(n295), .CI(n294), .CO(n383), .S(n379) );
  VHSR_AD1_1 U337 ( .A(n299), .B(n298), .CI(n297), .CO(n294), .S(n363) );
  VHSR_AOI31_2 U338 ( .A1(n303), .A2(n302), .A3(n301), .B(n300), .ZN(n362) );
  VHSR_AD1_1 U339 ( .A(n306), .B(n305), .CI(n304), .CO(n297), .S(n366) );
  VHSR_AOI22_2 U340 ( .A1(a[4]), .A2(b[5]), .B1(a[5]), .B2(b[4]), .ZN(n308) );
  VHSR_NOR2_1 U341 ( .A1(n308), .A2(n307), .ZN(n365) );
  VHSR_AD1_1 U342 ( .A(n311), .B(n310), .CI(n309), .CO(n304), .S(n369) );
  VHSR_AD1_1 U343 ( .A(n314), .B(n313), .CI(n312), .CO(n309), .S(n372) );
  VHSR_IN_2 U344 ( .I(n329), .ZN(n332) );
  VHSR_AD1_1 U345 ( .A(n317), .B(n316), .CI(n315), .CO(n331), .S(n208) );
  VHSR_AOI22_2 U346 ( .A1(b[3]), .A2(a[1]), .B1(b[1]), .B2(a[3]), .ZN(n330) );
  VHSR_OAI22_2 U347 ( .A1(n319), .A2(n322), .B1(n321), .B2(n318), .ZN(n320) );
  VHSR_OAI31_2 U348 ( .A1(n322), .A2(n321), .A3(n332), .B(n320), .ZN(n338) );
  VHSR_OAI211_2 U349 ( .A1(n329), .A2(n337), .B(a[3]), .C(b[3]), .ZN(n323) );
  VHSR_IN_2 U350 ( .I(n323), .ZN(n371) );
  VHSR_AD1_1 U351 ( .A(n326), .B(n325), .CI(n324), .CO(n312), .S(n377) );
  VHSR_CLKNAND2_2 U352 ( .A1(b[3]), .A2(a[3]), .ZN(n328) );
  VHSR_CLKNAND2_2 U353 ( .A1(n337), .A2(n328), .ZN(n327) );
  VHSR_OAI31_2 U354 ( .A1(n329), .A2(n337), .A3(n328), .B(n327), .ZN(n376) );
  VHSR_AD1_1 U355 ( .A(n332), .B(n331), .CI(n330), .CO(n339), .S(n391) );
  VHSR_NOR2_1 U356 ( .A1(n334), .A2(n333), .ZN(n335) );
  VHSR_AOI32_2 U357 ( .A1(b[0]), .A2(n336), .A3(a[4]), .B1(n335), .B2(n336), 
        .ZN(n390) );
  VHSR_AOI21_2 U358 ( .A1(n339), .A2(n338), .B(n337), .ZN(n374) );
  VHSR_AD1_1 U359 ( .A(n342), .B(n341), .CI(n340), .CO(n324), .S(n373) );
  VHSR_NOR2_1 U360 ( .A1(n344), .A2(n343), .ZN(n356) );
  VHSR_CLKNAND2_2 U361 ( .A1(a[6]), .A2(b[7]), .ZN(n346) );
  VHSR_AOI21_2 U362 ( .A1(a[7]), .A2(b[6]), .B(n346), .ZN(n345) );
  VHSR_AOI31_2 U363 ( .A1(a[7]), .A2(n346), .A3(b[6]), .B(n345), .ZN(n347) );
  VHSR_IN_2 U364 ( .I(n347), .ZN(n348) );
  VHSR_MAOI222_2 U365 ( .A(n351), .B(n349), .C(n348), .ZN(n358) );
  VHSR_OAI21_2 U366 ( .A1(n351), .A2(n350), .B(n358), .ZN(n355) );
  VHSR_CLKXOR2_2 U367 ( .A1(n356), .A2(n355), .Z(n352) );
  VHSR_CLKNAND2_2 U368 ( .A1(n353), .A2(n352), .ZN(n386) );
  VHSR_OAI21_2 U369 ( .A1(n353), .A2(n352), .B(n386), .ZN(n354) );
  VHSR_CLKNAND2_2 U370 ( .A1(a[7]), .A2(b[7]), .ZN(n385) );
  VHSR_NOR2_1 U371 ( .A1(n356), .A2(n355), .ZN(n357) );
  VHSR_AND3_2 U372 ( .A1(n387), .A2(n359), .A3(n386), .Z(n360) );
  VHSR_NOR2_1 U373 ( .A1(n385), .A2(n360), .ZN(product[15]) );
  VHSR_AD1_1 U374 ( .A(n377), .B(n376), .CI(n375), .CO(n370), .S(product[6])
         );
  VHSR_AD1_1 U375 ( .A(n380), .B(n379), .CI(n378), .CO(n381), .S(product[11])
         );
  VHSR_AD1_1 U376 ( .A(n383), .B(n382), .CI(n381), .CO(n353), .S(
        \intadd_0/SUM[7] ) );
  VHSR_NOR2_1 U377 ( .A1(n385), .A2(n384), .ZN(n388) );
  VHSR_XOR3_2 U378 ( .A1(n388), .A2(n387), .A3(n386), .Z(product[14]) );
  VHSR_AOI21_2 U379 ( .A1(n391), .A2(n390), .B(n389), .ZN(product[4]) );
endmodule

