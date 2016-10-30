/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.layers.*;
import cz.pv021.neuralnets.functions.*;
import java.util.Arrays;

/**
 *
 * @author lukas
 */
public class MLP {
    
    public static void main(String[] args) {
        
        
        IInputLayer l0 = new InputLayer(4);
        ILayer l1 = new FullyConnectedLayer(10, new HyperbolicTangent());
        ILayer l2 = new OutputLayer(3, new Softmax());
        Layers.Connect(l0, l1);
        Layers.Connect(l1, l2);
        
        long seed = 123;
        l0.initializeWeights(seed);
        l1.initializeWeights(seed);
        l2.initializeWeights(seed);
        
        double[] input = { 0.5, 0.2, 0.6, 0.8};
        l0.clampInput(input);
        l0.forwardPass();
        l1.forwardPass();
        l2.forwardPass();
        System.out.println(Arrays.toString(l2.GetOutput()));
        
    }
}
