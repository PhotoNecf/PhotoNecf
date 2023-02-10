package examples.generators.IOT;

import circuit.eval.CircuitEvaluator;
import circuit.structure.CircuitGenerator;
import circuit.structure.Wire;
import examples.gadgets.IOT.Corr2Gadget;

public class Corr2CircuitGenerator extends CircuitGenerator {
    private Wire[] lhs_sobel;
    private Wire[] rhs_im;
    private Wire sobel_th_ratio;
    private Wire percent_th;
    private Wire im_th_ratio;

    private Wire n_total;
    private Wire sobel_th;
    private Wire im_th;
    private Wire jacard_index;
    private Wire ratio_diff;

    private Wire maxResult;

    private Wire passResult;
    public final static long LENGTH = 255;

    public Corr2CircuitGenerator(String circuitName){super(circuitName);}

    @Override
    protected void buildCircuit(){
        lhs_sobel = createInputWireArray((int)LENGTH);
        rhs_im = createInputWireArray((int)LENGTH);
        sobel_th_ratio = createInputWire("sobel_th_ratio");
        percent_th = createInputWire("percent_th");
        n_total = createInputWire("n_total");
        im_th_ratio = createInputWire("im_th_ratio");
        sobel_th = createInputWire("sobel_th");
        im_th = createInputWire("im_th");
        ratio_diff = createInputWire("ratio_diff");

        Corr2Gadget corr2Gadget = new Corr2Gadget(lhs_sobel,rhs_im,sobel_th_ratio,im_th_ratio,percent_th,n_total,sobel_th,im_th,ratio_diff,"Corr1Gadget");
        maxResult = corr2Gadget.getOutputWires()[0];
        passResult = corr2Gadget.getOutputWires()[1];

        makeOutput(maxResult,"output 0 or 1");
        makeOutput(passResult,"output 0 or 1");


    }
    @Override
    public void generateSampleInput(CircuitEvaluator circuitEvaluator){
        int[] lhs_sobel_val = new int[(int)LENGTH];
        int[] rhs_im_val = new int[(int)LENGTH];
        int sobel_th_ratio_val = 0;
        int percent_th_val = 0;
        int im_th_ratio_val = 0;
        int n_total_val = 0;
        int sobel_th_val = 0;
        int im_th_val = 0;
        int ratio_diff_val = 0;

        for (int j = 0;j<LENGTH;j++){
            circuitEvaluator.setWireValue(lhs_sobel[j],lhs_sobel_val[j]);
            circuitEvaluator.setWireValue(rhs_im[j],rhs_im_val[j]);
        }
        circuitEvaluator.setWireValue(sobel_th_ratio, sobel_th_ratio_val);
        circuitEvaluator.setWireValue(percent_th, percent_th_val);
        circuitEvaluator.setWireValue(n_total, n_total_val);
        circuitEvaluator.setWireValue(im_th_ratio, im_th_ratio_val);
        circuitEvaluator.setWireValue(sobel_th, sobel_th_val);
        circuitEvaluator.setWireValue(im_th, im_th_val);
        circuitEvaluator.setWireValue(ratio_diff, ratio_diff_val);

    }

    public static void main(String[] args) throws Exception {
        Corr2CircuitGenerator generator = new Corr2CircuitGenerator("Corr2 circuit");
        generator.generateCircuit();
        generator.evalCircuit();
        generator.prepFiles();
        generator.runLibsnark();
    }
}
