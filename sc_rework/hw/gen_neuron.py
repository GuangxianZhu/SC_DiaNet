"""Generate a Verilog SC neuron (XNOR multipliers + APC + Btanh FSM) for fan-in m.

python gen_neuron.py M R > neuron_M.v
"""
import sys, math

m, r = int(sys.argv[1]), int(sys.argv[2])
cw = max(1, math.ceil(math.log2(m + 1)))      # APC count width
sw = max(1, math.ceil(math.log2(r)))           # state width
print(f"""module sc_neuron_m{m} (
  input clk, input rst,
  input  [{m-1}:0] x,     // input bitstreams (one bit per cycle)
  input  [{m-1}:0] w,     // weight bitstreams
  output y
);
  wire [{m-1}:0] p = ~(x ^ w);          // bipolar multiply
  // accumulative parallel counter (popcount)
  integer i;
  reg [{cw-1}:0] cnt;
  always @* begin
    cnt = 0;
    for (i = 0; i < {m}; i = i + 1) cnt = cnt + p[i];
  end
  // Btanh: saturating up/down counter, step = 2*cnt - m, states [0, {r-1}]
  reg [{sw-1}:0] s;
  wire signed [{sw+cw+2}:0] nxt = $signed({{1'b0, s}}) + $signed({{1'b0, cnt, 1'b0}}) - {m};
  always @(posedge clk) begin
    if (rst) s <= {r//2};
    else if (nxt < 0) s <= 0;
    else if (nxt > {r-1}) s <= {r-1};
    else s <= nxt[{sw-1}:0];
  end
  assign y = (s >= {r//2});
endmodule""")
