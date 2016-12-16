package cz.pv021.neuralnets.dataset.udcorpora;

import cz.pv021.neuralnets.dataset.SimpleExample;
import java.util.Arrays;

/**
 * @author  Lukas Daubner
 * @since   2016-12-15
 * @version 2016-12-15
 */
public class UdSimpleExample implements SimpleExample<UdLanguage> {
    
    double[] attributes;
    UdLanguage exampleClass;
    
    public UdSimpleExample (double[] attributes, UdLanguage exampleClass) {
        this.attributes = attributes;
        this.exampleClass = exampleClass;
    }

    @Override
    public double[] getAttributes() {
        return attributes;
    }

    @Override
    public UdLanguage getExampleClass() {
        return exampleClass;
    }

    @Override
    public String toString() {
        return exampleClass.toString() + ":" + Arrays.toString(attributes);
    }    
}
