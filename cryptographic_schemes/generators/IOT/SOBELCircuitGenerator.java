package examples.generators.IOT;
import util.Util;
import java.math.BigInteger;
import java.util.Arrays;

import circuit.structure.Wire;
import circuit.config.Config;
import circuit.structure.CircuitGenerator;
import circuit.eval.CircuitEvaluator;
import circuit.structure.WireArray;

import examples.gadgets.blockciphers.SymmetricEncryptionCBCGadget;
import examples.gadgets.diffieHellmanKeyExchange.FieldExtensionDHKeyExchange;
import examples.gadgets.hash.MerkleTreePathGadget;
import examples.gadgets.hash.SubsetSumHashGadget;
import examples.gadgets.math.DotProductGadget;
import examples.gadgets.hash.SHA256Gadget;
import examples.gadgets.IOT.SOBELGadget;
import org.bouncycastle.pqc.math.linearalgebra.IntegerFunctions;

public class SOBELCircuitGenerator extends CircuitGenerator {
    private Wire[] expandedInputWires;//expanded image for convolution ROWS*COLUMNS+2*(ROWS+COLUMNS)+4 elements
    private Wire[] convolutionResults;//ROWS*COLUMNS elements
    // private Wire[] results;
    // private Wire[] HConvResults;
    // private Wire[] VConvResults;
    public final static int ROWS = 255;
    public final static int COLUMNS = 255;
    // private Wire NUM2;

    public SOBELCircuitGenerator(String circuitName){
        super(circuitName);
    }

    @Override
    protected void buildCircuit(){
        // expandedInputWires = new Wire[(ROWS+2)*(COLUMNS+2)];
        expandedInputWires = createInputWireArray((ROWS+2)*(COLUMNS+2), "Inputwire array");
        // convolutionResults = createInputWireArray((ROWS)*(COLUMNS), "result array");

        // NUM2 = createInputWire();
        SOBELGadget iotGadget = new SOBELGadget(expandedInputWires, "");
        convolutionResults = iotGadget.getOutputWires();
        makeOutputArray(convolutionResults, "the result matrix");
    }

    @Override
    public void generateSampleInput(CircuitEvaluator circuitEvaluator) {
        // for (int i=1; i<=ROWS;i++ ) {
        //   for (int j =1;i<=COLUMNS ;j++ ) {
        //     circuitEvaluator.setWireValue(expandedInputWires[(i*(COLUMNS+2)+j],255);
        //   }
        // }
        //
        // for (int j = 1;j<=COLUMNS ;j++ ) {
        //   circuitEvaluator.setWireValue(expandedInputWires[j],255);
        //   circuitEvaluator.setWireValue(expandedInputWires[(ROWS+1)*(COLUMNS+2)+j],255);
        // }
        //
        // for (int i=1;i<=ROWS ;i++ ) {
        //   circuitEvaluator.setWireValue(expandedInputWires[i*(COLUMNS+2)],255);
        //   circuitEvaluator.setWireValue(expandedInputWires[i*(COLUMNS+2)+COLUMNS+1],255);
        // }
        //
        // circuitEvaluator.setWireValue(expandedInputWires[0],255);
        // circuitEvaluator.setWireValue(expandedInputWires[COLUMNS+1],255);
        // circuitEvaluator.setWireValue(expandedInputWires[ROWS+1*(COLUMNS+2)],255);
        // circuitEvaluator.setWireValue(expandedInputWires[ROWS+1*(COLUMNS+2)+COLUMNS+1],255);
        // circuitEvaluator.setWireValue(NUM2,2);
        for (int i = 0;i<ROWS+2 ; i++) {
            for (int j = 0;j<COLUMNS+2 ;j++ ) {
                circuitEvaluator.setWireValue(expandedInputWires[i*(COLUMNS+2)+j],255);
            }
        }
    }

    public static void main(String[] args) throws Exception {

        SOBELCircuitGenerator generator = new SOBELCircuitGenerator("IOTcircuit");
        generator.generateCircuit();
        generator.evalCircuit();
        generator.prepFiles();
        generator.runLibsnark();
    }
}
