package examples.gadgets.IOT;

import circuit.operations.Gadget;
import circuit.structure.Wire;

public class Corr2Gadget extends Gadget {

    private Wire[] lhs_sobel;
    private Wire[] rhs_im;
    private Wire sobel_th_ratio;
    private Wire percent_th;
    private Wire im_th_ratio;
    private Wire n_total;
    private Wire sobel_th;

    private Wire im_th;
    private Wire ratio_diff;
    private Wire maxResult;

    private Wire passResult;
    public final static long LENGTH = 255;

    public Corr2Gadget(Wire[] lhs_sobel, Wire[] rhs_im, Wire sobel_th_ratio,Wire im_th_ratio, Wire percent_th, Wire n_total, Wire sobel_th, Wire im_th, Wire ratio_diff, String desc){
        super(desc);
        this.lhs_sobel=lhs_sobel;
        this.rhs_im=rhs_im;
        this.sobel_th_ratio=sobel_th_ratio;
        this.percent_th=percent_th;
        this.sobel_th = sobel_th;
        this.im_th = im_th;
        this.n_total = n_total;
        this.ratio_diff = ratio_diff;
        this.im_th_ratio = im_th_ratio;
        buildCircuit();
    }

    private void buildCircuit(){
        Wire lhs_th_result;
        Wire im_th_result;

        lhs_th_result = check_lrhs_th_Gadget(lhs_sobel,sobel_th_ratio,sobel_th);
        im_th_result = check_lrhs_th_Gadget(rhs_im,im_th_ratio,im_th);

        Wire[] lhs;
        Wire[] rhs;
        lhs = threshodingGadget(lhs_sobel,sobel_th);
        rhs = threshodingGadget(rhs_im,im_th);

        Wire n_lhs;
        Wire n_rhs;
        n_lhs = countNonZeroGadget(lhs);
        n_rhs = countNonZeroGadget(rhs);

        Wire min_cover = n_total.mul(percent_th);
        Wire n_diff;
        Wire[] logicalXOR;
        logicalXOR = logicalXORGadget(lhs,rhs);
        n_diff = countNonZeroGadget(logicalXOR);

        Wire n_diff_check;
        n_diff_check = nDiffCheckGadget(n_diff,n_total,ratio_diff);

        maxResult = findMax(ratio_diff);

        passResult = passCheckGadget(n_diff_check,lhs_th_result,im_th_result,n_lhs,n_rhs,min_cover);


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

    private Wire[] logicalXORGadget(Wire[] lhs_, Wire[] rhs_){
        Wire[] result = generator.generateZeroWireArray((int)LENGTH);
        for (int i = 0;i<LENGTH;i++){
            result[i] = lhs_[i].xor(rhs_[i]);
        }
        return result;
    }
    private Wire nDiffCheckGadget(Wire n_diff_,Wire n_total_, Wire ratio_diff_){
       Wire result;
       Wire left = ratio_diff_.mul(n_total_);
       Wire right= n_diff_.mul(10000);
       Wire upperBond = right.add(100000000);
       Wire lowerBond = right.sub(100000000);
       Wire upperCmp = left.isLessThanOrEqual(upperBond,64);
       Wire lowerCmp = lowerBond.isLessThanOrEqual(left,64);
       result = upperCmp.mul(lowerCmp);
       return result;

    }

    private Wire findMax(Wire value){
        Wire temp = generator.createConstantWire(10000);
        temp = temp.sub(value);
        Wire cmp = temp.isGreaterThan(value,64);
        Wire OmCmp = generator.getOneWire();
        OmCmp = OmCmp.sub(cmp);
        Wire result = cmp.mul(temp).add(OmCmp.mul(value));
        return result;
    }
//    passResult = passCheckGadget(n_diff_check,lhs_th_result,im_th_result,n_lhs,n_rhs,min_cover);

    private Wire passCheckGadget(Wire n_diff_ckeck_,Wire lhs_th_result_,Wire im_th_result,Wire n_lhs_, Wire n_rhs_,Wire min_cover_){
        Wire temp1 = n_lhs_.isEqualTo(0);
        Wire temp2 = n_rhs_.isEqualTo(0);
        Wire temp3 = n_rhs_.isLessThan(min_cover_,64);
        Wire temp4 = n_lhs_.isLessThan(min_cover_,64);
        Wire tempAdd =n_lhs_.add(min_cover_);
        Wire temp5 = tempAdd.isGreaterThan(n_total,64);
        Wire temp6 = n_diff_ckeck_.isEqualTo(0);
        Wire temp7 = lhs_th_result_.isEqualTo(0);
        Wire temp8 = im_th_result.isEqualTo(0);


        temp8 = temp1.add(temp2).add(temp3).add(temp4).add(temp5).add(temp6).add(temp7).add(temp8);
        Wire result = temp8.isEqualTo(0);
        return result;
    }

    @Override
    public Wire[] getOutputWires() {
        Wire[] ret = new Wire[2];
        ret[0] = maxResult;
        ret[1] = passResult;

        return ret;
    }

}

