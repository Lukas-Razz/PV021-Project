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
public class Layers {
    
    //Na propojení vrstev
    public static void Connect(ILayer upper, ILayer lower)
    {
        upper.SetLowerLayer(lower);
        lower.SetUpperLayer(upper);
    }
    
}
