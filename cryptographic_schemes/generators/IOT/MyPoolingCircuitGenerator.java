package examples.generators.IOT;

import circuit.eval.CircuitEvaluator;
import circuit.structure.CircuitGenerator;
import circuit.structure.Wire;
import examples.gadgets.IOT.MyPoolingGadget;

public class MyPoolingCircuitGenerator extends CircuitGenerator {
    private Wire[][] im;
    private Wire[][] outP;
    private Wire result;
    public final static long STRIDE = 8;
    public final static long SIZE = 16;
    public final static long WIDE = 255;
    public final static long HEIGHT = 255;
    public final static long HALF_SIZE = 8;
    public final static long WIDENUM = 255;
    public final static long HEIGHTNUM = 255;

    public MyPoolingCircuitGenerator(String circuitName){super(circuitName);}

    @Override
    protected void buildCircuit(){
        for (int i = 0;i<HEIGHT;i++){
            im[i] = createInputWireArray((int)WIDE);
            outP[i] = createInputWireArray((int)WIDE);
        }


        MyPoolingGadget myPoolingGadget = new MyPoolingGadget(im,outP,"MyPoolingGadget");
        result = myPoolingGadget.getOutputWires()[0];

        makeOutput(result,"output 0 or 1");


    }
    @Override
    public void generateSampleInput(CircuitEvaluator circuitEvaluator){
        int[][] im_val = new int[(int)WIDE][(int)HEIGHT];
        int[][] outP_val = new int[(int)WIDE][(int)HEIGHT];


        for (int i = 0;i<WIDE;i++){
            for (int j = 0;j<HEIGHT;j++){
                circuitEvaluator.setWireValue(im[i][j],im_val[i][j]);
                circuitEvaluator.setWireValue(outP[i][j],outP_val[i][j]);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        MyPoolingCircuitGenerator generator = new MyPoolingCircuitGenerator("MyPooling circuit");
        generator.generateCircuit();
        generator.evalCircuit();
        generator.prepFiles();
        generator.runLibsnark();
    }
}
