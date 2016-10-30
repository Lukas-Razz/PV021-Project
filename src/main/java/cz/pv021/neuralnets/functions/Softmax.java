/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.pv021.neuralnets.functions;

/**
 *
 * @author lukas
 */
public class Softmax extends OutputFunction {

    @Override
    public double[] Function(double[] innerPotencials) {
        double[] parts = new double[innerPotencials.length];
        double sumOfParts = 0;
        double[] output = new double[innerPotencials.length];
        
        for(int i=0; i<innerPotencials.length; i++) {
            parts[i] = Math.pow(Math.E, innerPotencials[i]);
        }
        for(int i=0; i<parts.length; i++) {
            sumOfParts += parts[i];
        }
        for(int i=0; i<parts.length; i++) {
            output[i] = parts[i]/sumOfParts;
        }
        return output;
    }

    @Override
    public double[] Derivative(double[] innerPotencials) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
