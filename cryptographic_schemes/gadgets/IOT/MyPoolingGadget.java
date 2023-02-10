package examples.gadgets.IOT;

import circuit.operations.Gadget;
import circuit.structure.Wire;

public class MyPoolingGadget extends Gadget {

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


    public MyPoolingGadget(Wire[][] im, Wire[][] outP,String desc){
        super(desc);
        this.im = im;
        this.outP = outP;
        buildCircuit();
    }

    private void buildCircuit(){
        Wire meanCheckPass = generator.getZeroWire();
        for (int idxi = 0; idxi<WIDENUM;idxi++){
            for (int idxj = 0;idxj<HEIGHTNUM;idxj++){
                meanCheckPass = meanCheckPass.add(checkWindowMean(im,idxi,idxj,outP));
            }
        }

        result = meanCheckPass.isEqualTo(WIDENUM*HEIGHTNUM);

    }
    private Wire checkWindowMean(Wire[][] im_,int idxi_, int idxj_, Wire[][] outP_){
        Wire sum = generator.getZeroWire();
        for (int row =(int) (idxi_-HALF_SIZE);row<=(int)(idxi_+HALF_SIZE);row++){
            for (int col =(int) (idxj_-HALF_SIZE);col<=(int)(idxj_+HALF_SIZE);col++){
                sum = sum.add(im[row][col]);
            }
        }
        Wire left = outP_[idxi_][idxj_].mul(WIDENUM*HEIGHTNUM);
        Wire upperBond = left.add(10000);
        Wire lowerBond = left.sub(10000);
        Wire upperCmp = sum.isLessThanOrEqual(upperBond,64);
        Wire lowerCmp = lowerBond.isLessThanOrEqual(sum,64);

        Wire res = upperCmp.mul(lowerCmp);

        return res;
    }

    @Override
    public Wire[] getOutputWires() {
        Wire[] ret = new Wire[2];
        ret[0] = result;
        ret[1] = result;

        return ret;
    }

}

