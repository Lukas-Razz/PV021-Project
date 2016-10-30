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
public class HyperbolicTangent extends ActivationFunction {

    @Override
    public double Function(double innerPotencial) {
        return Math.tanh(innerPotencial);
    }

    @Override
    public double Derivative(double innerPotencial) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
