package examples.gadgets.IOT;

import circuit.operations.Gadget;
import circuit.structure.Wire;

public class Corr1Gadget extends Gadget {

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

    public Corr1Gadget(Wire[] lhs_sobel, Wire[] rhs_sobel, Wire sobel_th_ratio, Wire percent_th, Wire n_total, Wire lhs_th, Wire rhs_th, Wire jacard_index, String desc){
        super(desc);
        this.lhs_sobel=lhs_sobel;
        this.rhs_sobel=rhs_sobel;
        this.sobel_th_ratio=sobel_th_ratio;
        this.percent_th=percent_th;
        this.lhs_th = lhs_th;
        this.rhs_th = rhs_th;
        this.n_total = n_total;
        this.jacard_index = jacard_index;
        buildCircuit();
    }

    private void buildCircuit(){
        Wire lhs_th_result;
        Wire rhs_th_result;

        lhs_th_result = check_lrhs_th_Gadget(lhs_sobel,sobel_th_ratio,lhs_th);
        rhs_th_result = check_lrhs_th_Gadget(rhs_sobel,sobel_th_ratio,rhs_th);

        Wire[] lhs;
        Wire[] rhs;
        lhs = threshodingGadget(lhs_sobel,lhs_th);
        rhs = threshodingGadget(rhs_sobel,rhs_th);

        Wire n_lhs;
        Wire n_rhs;
        n_lhs = countNonZeroGadget(lhs);
        n_rhs = countNonZeroGadget(rhs);

        Wire min_cover = n_total.mul(percent_th);
        Wire n_intersec;
        Wire[] logicalAnd;
        logicalAnd = logicalAndGadget(lhs,rhs);
        n_intersec = countNonZeroGadget(logicalAnd);

        finalResult = jacardCheckGadget(n_intersec,n_lhs,n_rhs,jacard_index);
        passResult = passCheckGadget(lhs_th_result,rhs_th_result,min_cover,n_lhs,n_rhs);

    }

    private Wire check_lrhs_th_Gadget(Wire[] lrhs_sobel,Wire sobel_th_ratio, Wire lhs_th){
        Wire result;
        Wire sum = generator.getZeroWire();
        Wire left;
        for (int i = 0;i<LENGTH;i++){
            sum = sum.add(lrhs_sobel[i]);
        }
        sum = sum.mul(sobel_th_ratio);
        left = lhs_th.mul(LENGTH).mul(10000);
        Wire upperBond = left.add(100000000);
        Wire lowerBond = left.sub(100000000);
        Wire upperCmp = sum.isLessThanOrEqual(upperBond,64);
        Wire lowerCmp = lowerBond.isLessThanOrEqual(sum,64);
        result = upperCmp.mul(lowerCmp);
        return result;
    }

    private Wire[] threshodingGadget(Wire[] lrhs_sobel, Wire lhs_th){
        Wire[] result = generator.generateZeroWireArray((int)LENGTH);
        for (int i = 0;i<LENGTH;i++){
            result[i] = lrhs_sobel[i].isGreaterThan(lhs_th,64);
        }
        return result;
    }

    private Wire countNonZeroGadget(Wire[] lrhs){
        Wire result = generator.getZeroWire();
        for (int i = 0;i<LENGTH;i++){
            result = result.add(lrhs[i].isEqualTo(1));
        }
        return result;
    }

    private Wire[] logicalAndGadget(Wire[] lhs_, Wire[] rhs_){
        Wire[] result = generator.generateZeroWireArray((int)LENGTH);
        for (int i = 0;i<LENGTH;i++){
            result[i] = lhs_[i].and(rhs_[i]);
        }
        return result;
    }

    private Wire jacardCheckGadget(Wire n_intersec_,Wire n_lhs_, Wire n_rhs_,Wire jacard_index_){
        Wire result;
        Wire left = n_lhs_.add(n_rhs_).sub(n_intersec_);
        left = left.mul(jacard_index_);

        Wire ninterScale = n_intersec_.mul(10000);
        Wire upperBond = ninterScale.add(100000000);
        Wire lowerBond = ninterScale.sub(100000000);
        Wire upperCmp = left.isLessThanOrEqual(upperBond,64);
        Wire lowerCmp = lowerBond.isLessThanOrEqual(left,64);
        result = upperCmp.mul(lowerCmp);

        return result;
    }

//    passCheckGadget(lhs_th_result,rhs_th_result,min_cover,n_lhs,n_rhs);

    private Wire passCheckGadget(Wire lhs_th_result_,Wire rhs_th_result_, Wire min_cover_,Wire n_lhs_, Wire n_rhs_){
        Wire temp1 = n_lhs_.isEqualTo(0);
        Wire temp2 = n_rhs_.isEqualTo(0);
        Wire temp3 = n_rhs_.isLessThan(min_cover_,64);
        Wire temp4 = n_lhs_.isLessThan(min_cover_,64);
        Wire temp5 = lhs_th_result_.isEqualTo(0);
        Wire temp6 = rhs_th_result_.isEqualTo(0);

        temp6 = temp1.add(temp2).add(temp3).add(temp4).add(temp5).add(temp6);
        Wire result = temp6.isEqualTo(0);
        return result;
    }


        @Override
    public Wire[] getOutputWires() {
        Wire[] ret = new Wire[2];
        ret[0] = passResult;
        ret[1] = finalResult;

        return ret;
    }

}

