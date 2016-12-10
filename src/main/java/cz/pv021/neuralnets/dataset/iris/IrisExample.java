package cz.pv021.neuralnets.dataset.iris;

import cz.pv021.neuralnets.dataset.Example;
import cz.pv021.neuralnets.utils.Pair;

/**
 * @author  Josef Plch
 * @since   2016-12-08
 * @version 2016-12-10
 */
public class IrisExample extends Pair <double[], IrisClass> implements Example {
    public IrisExample (double[] attributes, IrisClass irisClass) {
        super (attributes, irisClass);
    }
    
    /**
     * Use getAttributes() instead.
     * 
     * @return Attributes.
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
     * Use getIrisClass() instead.
     * 
     * @return Iris class.
     */
    @Deprecated
    @Override
    public IrisClass getB () {
        return super.getB ();
    }
    
    @Override
    public IrisClass getIrisClass () {
        return super.getB ();
    }
}
