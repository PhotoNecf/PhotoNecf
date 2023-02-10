package examples.generators.IOT;

import circuit.structure.Wire;
import circuit.structure.CircuitGenerator;
import circuit.eval.CircuitEvaluator;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import examples.gadgets.IOT.NCCGadget;

public class NCCCircuitGenerator extends CircuitGenerator {
    public final static int ROWS = 512;
    public final static int COLUMNS = 512;
    private Wire[] inputRawPixels;
    private Wire[] inputNoisePixels;
    private Wire inputNormalValue;
    private Wire noiseNormalValue;
    private Wire nccValue;
    private Wire finalResult;

    public NCCCircuitGenerator(String circuitName){super(circuitName);}

    @Override
    protected void buildCircuit(){
        inputRawPixels = createInputWireArray(ROWS*COLUMNS);
        inputNoisePixels = createInputWireArray(ROWS*COLUMNS);
        inputNormalValue = createInputWire("input normal");
        noiseNormalValue = createInputWire("noise normal");
        nccValue = createInputWire("ncc value");

        NCCGadget nccGadget = new NCCGadget(inputRawPixels,inputNoisePixels,inputNormalValue,noiseNormalValue,nccValue,"ncc gadget");
        finalResult = nccGadget.getOutputWires()[0];
        makeOutput(finalResult,"output 0 or 1");


    }
    @Override
    public void generateSampleInput(CircuitEvaluator circuitEvaluator){
        int[] inputPxlInt = new int[ROWS*COLUMNS];
        int[] noisePxlInt = new int[ROWS*COLUMNS];
        int inputNorm = 0;
        int noiseNorm = 0;
        int cc = 0;
        int ncc = 0;
        try {
            File myObj = new File("Path to jsnark/JsnarkCircuitBuilder/src/examples/generators/IOT/writek1.txt");
            Scanner myReader = new Scanner(myObj);
            int i = 0;
            while (myReader.hasNextLine()) {
                inputPxlInt[i] = Integer.parseInt(myReader.nextLine());
                i++;
            }
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        try {
            File myObj = new File("Path to jsnark/JsnarkCircuitBuilder/src/examples/generators/IOT/writek2.txt");
            Scanner myReader = new Scanner(myObj);
            int i = 0;
            while (myReader.hasNextLine()) {
                noisePxlInt[i] = Integer.parseInt(myReader.nextLine());
                i++;
            }
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        try {
            File myObj = new File("Path to jsnark/JsnarkCircuitBuilder/src/examples/generators/IOT/writek1norm.txt");
            Scanner myReader = new Scanner(myObj);
            while (myReader.hasNextLine()) {
                float tempk1normfloat = Float.parseFloat(myReader.nextLine());
                inputNorm = (int)(tempk1normfloat*10000);
            }
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        try {
            File myObj = new File("Path to jsnark/JsnarkCircuitBuilder/src/examples/generators/IOT/writek2norm.txt");
            Scanner myReader = new Scanner(myObj);
            while (myReader.hasNextLine()) {
//                noiseNorm = Integer.parseInt(myReader.nextLine());
                float tempk2normfloat = Float.parseFloat(myReader.nextLine());
                noiseNorm = (int)(tempk2normfloat*10000);
            }
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        try {
            File myObj = new File("Path to jsnark/JsnarkCircuitBuilder/src/examples/generators/IOT/IOT/writecc.txt");
            Scanner myReader = new Scanner(myObj);
            while (myReader.hasNextLine()) {
//                cc = Integer.parseInt(myReader.nextLine());
                float tempccfloat = Float.parseFloat(myReader.nextLine());
                cc = (int)(tempccfloat*10000);
            }
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        try {
            File myObj = new File("Path to jsnark/JsnarkCircuitBuilder/src/examples/generators/IOT/writencc.txt");
            Scanner myReader = new Scanner(myObj);
            while (myReader.hasNextLine()) {
//                ncc = Integer.parseInt(myReader.nextLine());
                float tempnccfloat = Float.parseFloat(myReader.nextLine());
                ncc = (int)(tempnccfloat*10000);
                System.out.println("ncc is "+ ncc);

            }
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        for (int j = 0;j<ROWS*COLUMNS;j++){
            circuitEvaluator.setWireValue(inputRawPixels[j],inputPxlInt[j]);
            circuitEvaluator.setWireValue(inputNoisePixels[j],noisePxlInt[j]);
        }
        circuitEvaluator.setWireValue(inputNormalValue, inputNorm);
        circuitEvaluator.setWireValue(noiseNormalValue, noiseNorm);
        circuitEvaluator.setWireValue(nccValue, ncc);

    }

    public static void main(String[] args) throws Exception {
        NCCCircuitGenerator generator = new NCCCircuitGenerator("NCC circuit");
        generator.generateCircuit();
        generator.evalCircuit();
        generator.prepFiles();
        generator.runLibsnark();
    }
}
