package cz.pv021.neuralnets.layers;

/**
 * Just an interface alias.
 * 
 * @author  Josef Plch
 * @since   2016-11-08
 * @version 2016-11-27
 */
public interface OutputLayer extends LayerWithInput {
    
    void setExpectedOutput(double expectedOutput);
}
