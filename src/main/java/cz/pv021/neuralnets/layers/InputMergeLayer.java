package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.functions.HyperbolicTangent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Just a wrapper of two (paralell) layers.
 * 
 * @author  Josef Plch
 * @since   2016-12-11
 * @version 2016-12-12
 */
public class InputMergeLayer extends FullyConnectedLayer {
    private final LayerWithOutput inputLayerA;
    private final LayerWithOutput inputLayerB;
    private final int sizeA;
    private final int sizeB;
    
    // TODO: Activation function.
    // The function should not be used at all... Layer A & B functions should
    // be used instead.
    public InputMergeLayer (LayerWithOutput layerA, LayerWithOutput layerB) {
        super (
            layerA.getNumberOfUnits () + layerB.getNumberOfUnits (),
            null // new HyperbolicTangent ()
        );
        this.inputLayerA = layerA;
        this.inputLayerB = layerB;
        this.sizeA = layerA.getNumberOfUnits ();
        this.sizeB = layerB.getNumberOfUnits ();
    }
    
    public LayerWithOutput getLayerA () {
        return inputLayerA;
    }
    
    public LayerWithOutput getLayerB () {
        return inputLayerB;
    }
    
    @Override
    public void backwardPass () {
        copyAttributesFromChildren ();
        super.backwardPass ();
        copyAttributesToChildren ();
    }
    
    private void copyAttributesToChildren () {
        if (inputLayerA instanceof FullyConnectedLayer) {
            FullyConnectedLayer layerA = (FullyConnectedLayer) inputLayerA;
            layerA.setBias (partA (this.getBias ()));
            //layerA.setBiasErrors (partA (this.getBiasErrors ()));
            layerA.setErrWrtInnerP (partA (this.getErrWrtInnerP ()));
            layerA.setInnerPotentials (partA (this.getInnerPotentials ()));
            layerA.setOutput (partA (this.getOutput ()));
            //layerA.setWeightErrors (partA (this.getWeightErrors ()));
            layerA.setWeights (partA (this.getWeights ()));
        }
        
        if (inputLayerB instanceof FullyConnectedLayer) {
            FullyConnectedLayer layerB = (FullyConnectedLayer) inputLayerB;
            layerB.setBias (partB (this.getBias ()));
            //layerB.setBiasErrors (partB (this.getBiasErrors ()));
            layerB.setErrWrtInnerP (partB (this.getErrWrtInnerP ()));
            layerB.setInnerPotentials (partB (this.getInnerPotentials ()));
            layerB.setOutput (partB (this.getOutput ()));
            //layerB.setWeightErrors (partB (this.getWeightErrors ()));
            layerB.setWeights (partB (this.getWeights ()));
        }
    }
    
    private void copyAttributesFromChildren () {
        if (inputLayerA instanceof FullyConnectedLayer && inputLayerB instanceof FullyConnectedLayer) {
            FullyConnectedLayer layerA = (FullyConnectedLayer) inputLayerA;
            FullyConnectedLayer layerB = (FullyConnectedLayer) inputLayerB;
            
            this.setBias (merge (
                layerA.getBias (),
                layerB.getBias ()
            ));
            this.setBiasErrors (merge (
                layerA.getBiasErrors (),
                layerB.getBiasErrors ()
            ));
            this.setErrWrtInnerP (merge (
                layerA.getErrWrtInnerP (),
                layerB.getErrWrtInnerP ()
            ));
            this.setInnerPotentials (merge (
                layerA.getInnerPotentials (),
                layerB.getInnerPotentials ()
            ));
            this.setOutput (merge (
                layerA.getOutput (),
                layerB.getOutput ()
            ));
            this.setWeightErrors (merge (
                layerA.getWeightErrors (),
                layerB.getWeightErrors ()
            ));
            this.setWeights (merge (
                layerA.getWeights (),
                layerB.getWeights ()
            ));
        }
    }
    
    @Override
    public void forwardPass () {
        copyAttributesFromChildren ();
        super.forwardPass ();
        copyAttributesToChildren ();
    }
    
    private double[] merge (double[] a, double[] b) {
        double[] merged = new double[a.length + b.length];
        System.arraycopy (a, 0, merged, 0,        a.length);
        System.arraycopy (b, 0, merged, a.length, b.length);
        return merged;
    }
    
    private double[][] merge (double[][] a, double[][] b) {
        if (a[0].length != b[0].length) {
            int x1 = a[0].length;
            int x2 = b[0].length;
            int y1 = a.length;
            int y2 = b.length;
            throw new IllegalArgumentException (
                "Array shapes differ: "
                + x1 + "*" + y1 + " vs " + x2 + "*" + y2
            );
        }
        double[][] merged = new double[a.length + b.length][a[0].length];
        System.arraycopy (a, 0, merged, 0,        a.length);
        System.arraycopy (b, 0, merged, a.length, b.length);
        return merged;
    }
    
    private <E> List <E> merge (List <E> a, List <E> b) {
        List <E> merged = new ArrayList <> ();
        merged.addAll (a);
        merged.addAll (b);
        return merged;
    }
    
    private double[] partA (double[] array) {
        return Arrays.copyOfRange (array, 0, sizeA);
    }
    
    private double[][] partA (double[][] array) {
        return Arrays.copyOfRange (array, 0, sizeA);
    }
    
    private <E> List <E> partA (List <E> list) {
        return list.subList (0, sizeA);
    }
    
    private double[] partB (double[] array) {
        return Arrays.copyOfRange (array, sizeA, sizeA + sizeB);
    }
    
    private double[][] partB (double[][] array) {
        return Arrays.copyOfRange (array, sizeA, sizeA + sizeB);
    }
    
    private <E> List <E> partB (List <E> list) {
        return list.subList (sizeA, sizeA + sizeB);
    }
    
    @Override
    public void resetGradients () {
        copyAttributesFromChildren ();
        super.resetGradients ();
        copyAttributesToChildren ();
    }
    
    @Override
    public void setOutputLayer (LayerWithInput layer) {
        super.setOutputLayer (layer);
        inputLayerA.setOutputLayer (layer);
        inputLayerB.setOutputLayer (layer);
    }
}
