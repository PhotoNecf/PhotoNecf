package examples.generators.IOT;

import circuit.eval.CircuitEvaluator;
import circuit.structure.CircuitGenerator;
import circuit.structure.Wire;
import examples.gadgets.IOT.Corr1Gadget;

public class Corr1CircuitGenerator extends CircuitGenerator {
    private Wire[] lhs_sobel;
    private Wire[] rhs_sobel;
    private Wire sobel_th_ratio;
    private Wire percent_th;
    private Wire n_total;
    private Wire lhs_th;
    private Wire rhs_th;
    private Wire jacard_index;
    private Wire finalResult;

    private Wire passResult;
    public final static long LENGTH = 255;

    public Corr1CircuitGenerator(String circuitName){super(circuitName);}

    @Override
    protected void buildCircuit(){
        lhs_sobel = createInputWireArray((int)LENGTH);
        rhs_sobel = createInputWireArray((int)LENGTH);
        sobel_th_ratio = createInputWire("sobel_th_ratio");
        percent_th = createInputWire("percent_th");
        n_total = createInputWire("n_total");
        lhs_th = createInputWire("lhs_th");
        rhs_th = createInputWire("rhs_th");
        jacard_index = createInputWire("jacard_index");

        Corr1Gadget corr1Gadget = new Corr1Gadget(lhs_sobel,rhs_sobel,sobel_th_ratio,percent_th,n_total,lhs_th,rhs_th,jacard_index,"Corr1Gadget");
        passResult = corr1Gadget.getOutputWires()[0];
        finalResult = corr1Gadget.getOutputWires()[1];

        makeOutput(passResult,"output 0 or 1");
        makeOutput(finalResult,"output 0 or 1");


    }
    @Override
    public void generateSampleInput(CircuitEvaluator circuitEvaluator){
        int[] lhs_sobel_val = new int[(int)LENGTH];
        int[] rhs_sobel_val = new int[(int)LENGTH];
        int sobel_th_ratio_val = 0;
        int percent_th_val = 0;
        int n_total_val = 0;
        int lhs_th_val = 0;
        int rhs_th_val = 0;
        int jacard_index_val = 0;

        for (int j = 0;j<LENGTH;j++){
            circuitEvaluator.setWireValue(lhs_sobel[j],lhs_sobel_val[j]);
            circuitEvaluator.setWireValue(rhs_sobel[j],rhs_sobel_val[j]);
        }
        circuitEvaluator.setWireValue(sobel_th_ratio, sobel_th_ratio_val);
        circuitEvaluator.setWireValue(percent_th, percent_th_val);
        circuitEvaluator.setWireValue(n_total, n_total_val);
        circuitEvaluator.setWireValue(lhs_th, lhs_th_val);
        circuitEvaluator.setWireValue(rhs_th, rhs_th_val);
        circuitEvaluator.setWireValue(jacard_index, jacard_index_val);

    }

    public static void main(String[] args) throws Exception {
        Corr1CircuitGenerator generator = new Corr1CircuitGenerator("Corr1 circuit");
        generator.generateCircuit();
        generator.evalCircuit();
        generator.prepFiles();
        generator.runLibsnark();
    }
}
