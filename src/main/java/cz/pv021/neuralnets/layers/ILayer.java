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
public interface ILayer 
{
    //Při inferenci
    void forwardPass();
    //Při backpropagation
    void backwardPass();
    
    //Nastavení metodou Layers.Connect
    void SetUpperLayer(ILayer layer);
    void SetLowerLayer(ILayer layer);
    
    int GetNumberOfUnits();
    double[] GetOutput();
}
