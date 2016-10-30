/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.pv021.neuralnets.layers;

/**
 *
 * @author lukas
 */
public interface IInputLayer extends ILayer {
    
    public void clampInput(double[] input);
    
}
