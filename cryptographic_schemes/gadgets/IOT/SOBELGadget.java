package examples.gadgets.IOT;

import java.io.*;
import util.Util;
import circuit.operations.Gadget;
import circuit.config.Config;
import circuit.structure.Wire;
import circuit.structure.WireArray;
import java.math.BigInteger;
import java.util.Arrays;
import examples.gadgets.math.ModConstantGadget;
import examples.gadgets.math.FieldDivisionGadget;
import circuit.eval.Instruction;
import java.math.BigInteger;
import org.bouncycastle.pqc.math.linearalgebra.IntegerFunctions;
import circuit.eval.CircuitEvaluator;
import circuit.structure.ConstantWire;



public class SOBELGadget extends Gadget {
    private Wire[] expandedInputWires;//expanded image for convolution ROWS*COLUMNS+2*(ROWS+COLUMNS)+4 elements
    private Wire[] convolutionResults;//ROWS*COLUMNS elements
    // private Wire[] results;
    // private Wire[] HConvResults;
    // private Wire[] VConvResults;
    public final static int ROWS = 255;
    public final static int COLUMNS = 255;
    // private Wire NUM2;
    // public final static BigInteger NUM2BIG = 2;
    public final static String direct = "all";


    public SOBELGadget(Wire[] expandedInputWires, String desc){
        super(desc);
        this.expandedInputWires = expandedInputWires;
        buildCircuit();
    }

    private void buildCircuit(){
        Wire[] HConvResults = new Wire[ROWS*COLUMNS];
        Wire[] VConvResults = new Wire[ROWS*COLUMNS];
        if(direct.equals("all") || direct.equals("h")){
            HConvResults = convolutionHCircuit(expandedInputWires);
            HConvResults = matrixSquare(HConvResults);
            // convolutionResults = matrixAddCircuit(HConvResults,convolutionResults);
        }
        if(direct.equals("all") || direct.equals("v")){
            VConvResults = convolutionVCircuit(expandedInputWires);
            VConvResults = matrixSquare(VConvResults);
            // convolutionResults = matrixAddCircuit(VConvResults,convolutionResults);
        }
        convolutionResults = matrixAddCircuit(HConvResults,VConvResults);

    }

    private Wire[] matrixSquare(Wire[] inMatrix){
        Wire[] tempSquare = new Wire[ROWS*COLUMNS];
        for (int j=1;j<=COLUMNS ;j++ ) {
            for (int i=1;i<=ROWS ;i++ ) {
                tempSquare[(i-1)*COLUMNS+j-1] = inMatrix[(i-1)*COLUMNS+j-1].mul(inMatrix[(i-1)*COLUMNS+j-1]);
            }
        }
        return tempSquare;
    }

    private Wire[] convolutionHCircuit(Wire[] theInputWires){
        Wire[] tempH = new Wire[ROWS*COLUMNS];

        System.out.println("ROWS " + ROWS);
        System.out.print("COLUMNS " + COLUMNS);

        for (int j=1;j<=COLUMNS ;j++ ) {
            for (int i=1;i<=ROWS ;i++ ) {

                // tempH[i][j]=theInputWires[i-1][j-1].add(NUM2.mul(theInputWires[i-1][j])).add(theInputWires[i-1][j+1]).sub(theInputWires[i+1][j-1]).sub(NUM2.mul(theInputWires[i+1][j])).sub(theInputWires[i+1][j+1]);
                // tempH[(i-1)*COLUMNS+j-1]=theInputWires[(i-1)*(COLUMNS+2)+j-1].add(NUM2.mul(theInputWires[(i-1)*(COLUMNS+2)+j])).add(theInputWires[(i-1)*(COLUMNS+2)+j+1]).sub(theInputWires[(i+1)*(COLUMNS+2)+j-1]).sub(NUM2.mul(theInputWires[(i+1)*(COLUMNS+2)+j])).sub(theInputWires[(i+1)*(COLUMNS+2)+j+1]);
                tempH[(i-1)*COLUMNS+j-1]=theInputWires[(i-1)*(COLUMNS+2)+j-1].add(theInputWires[(i-1)*(COLUMNS+2)+j]).add(theInputWires[(i-1)*(COLUMNS+2)+j]).add(theInputWires[(i-1)*(COLUMNS+2)+j+1]).sub(theInputWires[(i+1)*(COLUMNS+2)+j-1]).sub(theInputWires[(i+1)*(COLUMNS+2)+j]).sub(theInputWires[(i+1)*(COLUMNS+2)+j]).sub(theInputWires[(i+1)*(COLUMNS+2)+j+1]);

            }
        }
        return tempH;
    }

    private Wire[] convolutionVCircuit(Wire[] theInputWires){
        Wire[] tempV = new Wire[ROWS*COLUMNS];
        for (int j=1;j<=COLUMNS ;j++ ) {
            for (int i=1;i<=ROWS ;i++ ) {
                // tempV[i][j] = expandedInputWires[i-1][j-1].add(NUM2.mul(expandedInputWires[i][j-1])).add(expandedInputWires[i+1][j-1]).sub(expandedInputWires[i-1][j+1]).sub(NUM2.mul(expandedInputWires[i][j+1])).sub(expandedInputWires[i+1][j+1]);
                tempV[(i-1)*COLUMNS+j-1]=theInputWires[(i-1)*(COLUMNS+2)+j-1].add(theInputWires[(i)*(COLUMNS+2)+j-1]).add(theInputWires[(i)*(COLUMNS+2)+j-1]).add(theInputWires[(i+1)*(COLUMNS+2)+j-1]).sub(theInputWires[(i-1)*(COLUMNS+2)+j+1]).sub(theInputWires[(i)*(COLUMNS+2)+j+1]).sub(theInputWires[(i)*(COLUMNS+2)+j+1]).sub(theInputWires[(i+1)*(COLUMNS+2)+j+1]);
            }
        }
        return tempV;
    }

    private Wire[] matrixAddCircuit(Wire[] convRes,Wire[] finalResult){
        Wire[] theResult = new Wire[ROWS*COLUMNS];

        for (int j=1;j<=COLUMNS ;j++ ) {
            for (int i=1;i<=ROWS ;i++ ) {
                theResult[(i-1)*COLUMNS+j-1] = convRes[(i-1)*COLUMNS+j-1].add(finalResult[(i-1)*COLUMNS+j-1]);
            }
        }
        return theResult;
    }
    @Override
    public Wire[] getOutputWires() {
        Wire[] result = new Wire[ROWS*COLUMNS];
        for (int i = 0; i<ROWS;i++ ) {
            for (int j=0;j<COLUMNS ;j++ ) {
                result[i*COLUMNS+j] = convolutionResults[i*COLUMNS+j];
            }
        }
        return result;
    }
}
