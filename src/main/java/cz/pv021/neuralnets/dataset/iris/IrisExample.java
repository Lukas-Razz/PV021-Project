package cz.pv021.neuralnets.dataset.iris;

import cz.pv021.neuralnets.dataset.DataClass;
import cz.pv021.neuralnets.dataset.Example;
import cz.pv021.neuralnets.utils.Pair;

/**
 * @author  Josef Plch
 * @since   2016-12-08
 * @version 2016-12-13
 */
public class IrisExample extends Pair <double[], DataClass> implements Example {
    public IrisExample (double[] attributes, DataClass exampleClass) {
        super (attributes, exampleClass);
    }
    
    /**
     * Use getAttributes() instead.
     * 
     * @return Example attributes.
     */
    @Deprecated
    @Override
    public double[] getA () {
        return super.getA ();
    }
    
    @Override
    public double[] getAttributes () {
        return super.getA ();
    }
    
    /**
     * Use getDataClass() instead.
     * 
     * @return Example class.
     */
    @Deprecated
    @Override
    public DataClass getB () {
        return super.getB ();
    }
    
    @Override
    public DataClass getExampleClass () {
        return super.getB ();
    }
}
