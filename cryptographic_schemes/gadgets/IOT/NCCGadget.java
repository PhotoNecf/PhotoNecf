package examples.gadgets.IOT;
import java.io.*;

import circuit.structure.BitWire;
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
public class NCCGadget extends Gadget {
    public final static int ROWS = 512;
    public final static int COLUMNS = 512;

    private Wire[] inputRawPixels;
    private Wire[] inputNoisePixels;
    private Wire inputNormalValue;
    private Wire noiseNormalValue;
    private Wire nccValue;
    private Wire finalResult;

//to delete 3


    public NCCGadget(Wire[] inputRawPixels, Wire[] inputNoisePixels, Wire inputNormalValue, Wire noiseNormalValue, Wire nccValue, String desc){
        super(desc);
        this.inputNoisePixels = inputNoisePixels;
        this.inputRawPixels = inputRawPixels;
        this.inputNormalValue = inputNormalValue;
        this.noiseNormalValue = noiseNormalValue;
        this.nccValue = nccValue;
        buildCircuit();
    }

    private void buildCircuit(){
        Wire sumInputRaw;
        Wire sumInputNoise;
        Wire cc;
        Wire inputNormPass;
        Wire noiseNormPass;
        Wire nccCheckPass;

        Wire normProduct;

//        normalizeCircuit();
        sumInputRaw = sumGadget(inputRawPixels);
        inputNormPass = normCheckGadget(inputNormalValue,sumInputRaw);

        sumInputNoise = sumGadget(inputNoisePixels);
        noiseNormPass = normCheckGadget(noiseNormalValue,sumInputNoise);

        cc = arrayProductGadget(inputRawPixels,inputNoisePixels);
        nccCheckPass = nccCheckGadget(nccValue,cc,inputNormalValue,noiseNormalValue);
        finalResult = finalCheckGadget(nccCheckPass,inputNormPass,noiseNormPass);

    }

    private Wire sumGadget(Wire[] inputArray){
        Wire sum = generator.getZeroWire();
        for (int i = 0; i<COLUMNS*ROWS;i++){
            sum = sum.add(inputArray[i]);
        }
        return sum;
    }

    private Wire normCheckGadget(Wire normValue, Wire sumValue){
        Wire sumScale = sumValue.mul(100000000);
        Wire normSquare = normValue.mul(normValue);
        Wire normSquareUpper = normSquare.add(100000000);
        Wire normSquareLower = normSquare.sub(100000000);
        Wire upperCmp = sumScale.isLessThanOrEqual(normSquareUpper,64);
        Wire lowerCmp = normSquareLower.isLessThanOrEqual(sumScale,64);
        Wire isInInterval = upperCmp.mul(lowerCmp);
        return isInInterval;
    }

    private Wire arrayProductGadget(Wire[] arrayA, Wire[] arrayB){
        Wire sum = generator.getZeroWire();
        for (int i = 0;i<COLUMNS*ROWS;i++){
            sum = sum.add(((arrayA[i]).mul(arrayB[i])));
        }
        return sum;
    }

    private Wire nccCheckGadget(Wire nccValue, Wire cc, Wire inputNormalValue, Wire noiseNormalValue){
        Wire productTemp = inputNormalValue.mul(noiseNormalValue).mul(nccValue);
        //norm value %.4d, ncc value %.4d
        Wire ccScale = cc.mul(100000000).mul(10000);
        Wire tempOneWire = generator.getOneWire().mul(300000000).mul(10000);
        Wire ccScaleUpper = ccScale.add(tempOneWire);
        Wire ccScaleLower = ccScale.sub(tempOneWire);
        Wire upperCmp = productTemp.isLessThanOrEqual(ccScaleUpper,64);
        Wire lowerCmp = ccScaleLower.isLessThanOrEqual(productTemp,64);
        Wire isInInterval = upperCmp.mul(lowerCmp);
        return isInInterval;
    }

    private Wire finalCheckGadget(Wire passA, Wire passB, Wire passC){
        Wire ret = passA.mul(passB).mul(passC);
        return ret;
    }

    @Override
    public Wire[] getOutputWires() {
        Wire[] ret = new Wire[2];
        ret[0] = finalResult;
        ret[1] = finalResult;

        return ret;
    }

}

