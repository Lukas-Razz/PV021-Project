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
public abstract class ActivationFunction {
    public abstract double Function(double innerPotencial);
    public abstract double Derivative(double innerPotencial);
}
