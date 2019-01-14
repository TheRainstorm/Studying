package JFrame;
import javax.swing.JButton;
import javax.swing.JFrame;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Toolkit;


public class ShowFlowLayout extends JFrame {
	public ShowFlowLayout() {
		//super("ShowFlowLayout");还是在main中设置的风格好些
		
		//get the content pane of the frame
		Container container=getContentPane();
		
		//set FlowLayout, aligned left, horizontal gap 10
		//vertical gap 20
		container.setLayout(new FlowLayout(FlowLayout.LEFT,10,20));
		
		//Add buttons to the frame;
		for(int i=0;i<10;i++) {
			container.add(new JButton("Componet "+(i+1)));
		}
	}
	
	/**Main Method*/
	public static void main(String[] args) {
		ShowFlowLayout frame=new ShowFlowLayout();
		frame.setTitle("ShowFlowLayout");
		frame.setDefaultCloseOperation(EXIT_ON_CLOSE);
		frame.setSize(500,200);
		
		//get the screen size
		Dimension screenSize=Toolkit.getDefaultToolkit().getScreenSize();
		int screenWidth=screenSize.width;
		int screenHeight=screenSize.height;
		//get x,y
		int x=(screenWidth-frame.getWidth())/2;
		int y=(screenHeight-frame.getHeight())/2;
		//set the frame to the screen center
		frame.setLocation(x,y);
		
		frame.setVisible(true);

	}

}
