package cz.pv021.neuralnets.dataset;

/**
 * @param <A> Type of attributes.
 * @param <C> Type of data class.
 *
 * @author  Josef Plch
 * @since   2016-12-15
 * @version 2016-12-15
 */
public interface Example <A, C extends DataClass> {
    public A getAttributes ();
    
    public C getExampleClass ();
}
