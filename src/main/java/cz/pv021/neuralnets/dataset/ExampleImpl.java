package cz.pv021.neuralnets.dataset;

import cz.pv021.neuralnets.utils.Pair;

/**
 * @param <C> Type of data class.
 * 
 * @author  Josef Plch
 * @since   2016-12-15
 * @version 2016-12-15
 */
public class ExampleImpl <A, C extends DataClass> extends Pair <A, C> implements Example <A, C> {
    public ExampleImpl (A attributes, C exampleClass) {
        super (attributes, exampleClass);
    }
    
    /**
     * Use getAttributes() instead.
     * 
     * @return Example attributes.
     */
    @Deprecated
    @Override
    public A getA () {
        return super.getA ();
    }
    
    @Override
    public A getAttributes () {
        return super.getA ();
    }
    
    /**
     * Use getDataClass() instead.
     * 
     * @return Example class.
     */
    @Deprecated
    @Override
    public C getB () {
        return super.getB ();
    }
    
    @Override
    public C getExampleClass () {
        return super.getB ();
    }
}
