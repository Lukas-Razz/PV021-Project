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
public abstract class OutputFunction {
    public abstract double[] Function(double[] innerPotencials);
    public abstract double[] Derivative(double[] innerPotencials);
}
