#include "crossbar_models.h"

static const char fixed_resistor_netlist[] =
	"* Fixed resistor memory-device baseline\n"
	"* Two-terminal, non-stateful element\n"
	"\n"
	".subckt fixed_resistor plus minus PARAMS: Rinit=80k\n"
	"\n"
	"Rdevice plus minus {Rinit}\n"
	"\n"
	".ends fixed_resistor\n";

static const char pershin_netlist[] =
	".subckt memristor pl mn PARAMS: Ron=1K Roff=10K Rinit=5K "
	"alpha=0 beta=1E13 Vt=4.6\n"
	"Bx 0 x I='(f1(V(pl,mn))>0) && (V(x)<Roff) ? {f1(V(pl,mn))}: "
	"(f1(V(pl,mn))<0) && (V(x)>Ron) ? {f1(V(pl,mn))}: {0}'\n"
	"Cx x 0 1 IC={Rinit}\n"
	"R0 pl mn 1E12\n"
	"Rmem pl mn r={V(x)}\n"
	".func f1(y)={beta*y+0.5*(alpha-beta)*(abs(y+Vt)-abs(y-Vt))}\n"
	".ends\n";

static const char hp_netlist[] =
	"* HP Memristor SPICE Model\n"
	"* For Transient Analysis only\n"
	"* created by Zdenek and Dalibor Biolek\n"
	"**************************\n"
	"* Ron, Roff - Resistance in ON / OFF States\n"
	"* Rinit - Resistance at T=0\n"
	"* D - Width of the thin film\n"
	"* uv - Migration coefficient\n"
	"* p - Parameter of the WINDOW-function\n"
	"* for modeling nonlinear boundary conditions\n"
	"* x - W/D Ratio, W is the actual width\n"
	"* of the doped area (from 0 to D)\n"
	"*\n"
	".SUBCKT memristor Plus Minus PARAMS:\n"
	"+ Ron=1K Roff=100K Rinit=80K D=10N uv=10F p=1\n"
	"***********************************************\n"
	"* DIFFERENTIAL EQUATION MODELING *\n"
	"***********************************************\n"
	"Gx 0 x value={ I(Emem)*uv*Ron/D^2*f(V(x),p)}\n"
	"Cx x 0 1 IC={(Roff-Rinit)/(Roff-Ron)}\n"
	"Raux x 0 1T\n"
	"\n"
	"* RESISTIVE PORT OF THE MEMRISTOR *\n"
	"*******************************\n"
	"Emem plus aux value={-I(Emem)*V(x)*(Roff-Ron)}\n"
	"Roff aux minus {Roff}\n"
	"***********************************************\n"
	"*Flux computation*\n"
	"***********************************************\n"
	"*does not work with ngspice\n"
	"*Eflux flux 0 value={SDT(V(plus,minus))}\n"
	"***********************************************\n"
	"*Charge computation*\n"
	"***********************************************\n"
	"*does not work with ngspice\n"
	"*Echarge charge 0 value={SDT(I(Emem))}\n"
	"***********************************************\n"
	"* WINDOW FUNCTIONS\n"
	"* FOR NONLINEAR DRIFT MODELING *\n"
	"***********************************************\n"
	"*window function, according to Joglekar\n"
	".func f(x,p)={1-(2*x-1)^(2*p)}\n"
	"*proposed window function\n"
	";.func f(x,i,p)={1-(x-stp(-i))^(2*p)}\n"
	".ENDS memristor\n";

static const char yakopcic_netlist[] =
	".subckt MEM_YAKOPCIC TE BE params:\n"
	"+ Rinit=1000\n"
	"+ a1=0.17 a2=0.17 b=0.05\n"
	"+ Vp=0.65 Vn=0.56\n"
	"+ Ap=4000 An=4000\n"
	"+ xp=0.3 xn=0.5\n"
	"+ alphap=1 alphan=5\n"
	"+ eta=1\n"
	"\n"
	".param xo={1/(a1*b*Rinit)}\n"
	"\n"
	".func wp(x)={xp/(1-xp)-x/(1-xp)+1}\n"
	".func wn(x)={x/(1-xn)}\n"
	"\n"
	".func G(v)={ternary_fcn(v<=Vp,ternary_fcn(v>=-Vn,0,"
	"-An*(exp(-v)-exp(Vn))),Ap*(exp(v)-exp(Vp)))}\n"
	"\n"
	".func F(v,x)={ternary_fcn(eta*v>=0,ternary_fcn(x>=xp,"
	"exp(-alphap*(x-xp))*wp(x),1),ternary_fcn(x<=(1-xn),"
	"exp(alphan*(x+xn-1))*wn(x),1))}\n"
	"\n"
	".func IVRel(v,x)={ternary_fcn(v>=0,a1*x*sinh(b*v),"
	"a2*x*sinh(b*v))}\n"
	"\n"
	"Cx xsv 0 1 IC={xo}\n"
	"Rx xsv 0 1T\n"
	"\n"
	"Gx 0 xsv value={\n"
	"+ eta*F(V(TE,BE),V(xsv))*G(V(TE,BE))\n"
	"+ }\n"
	"\n"
	"Gm TE BE value={\n"
	"+ IVRel(V(TE,BE),V(xsv))\n"
	"+ }\n"
	"\n"
	".ends MEM_YAKOPCIC\n";

static const struct crossbar_model_definition fixed_resistor_model = {
	.subcircuit_name = "fixed_resistor",
	.netlist = fixed_resistor_netlist,
};

static const struct crossbar_model_definition pershin_model = {
	.subcircuit_name = "memristor",
	.netlist = pershin_netlist,
};

static const struct crossbar_model_definition hp_model = {
	.subcircuit_name = "memristor",
	.netlist = hp_netlist,
};

static const struct crossbar_model_definition yakopcic_model = {
	.subcircuit_name = "MEM_YAKOPCIC",
	.netlist = yakopcic_netlist,
};

const struct crossbar_model_definition *
crossbar_model_get(spires_crossbar_model model)
{
	switch (model) {
	case SPIRES_CROSSBAR_MODEL_YAKOPCIC:
		return &yakopcic_model;
	case SPIRES_CROSSBAR_MODEL_PERSHIN:
		return &pershin_model;
	case SPIRES_CROSSBAR_MODEL_HP:
		return &hp_model;
	case SPIRES_CROSSBAR_MODEL_FIXED_R:
		return &fixed_resistor_model;
	default:
		return NULL;
	}
}
