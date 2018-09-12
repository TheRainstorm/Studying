package JFrame;
import javax.swing.JButton;
import javax.swing.JFrame;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.Toolkit;

public class ShowGridLayout extends JFrame{
	public ShowGridLayout() {
		//G et the content pane of the frame
		Container container=getContentPane();
		
		//Set GridLayout, 2 rows, 3 columns, and gaps 5 between
		//components horizontally and vertically
		container.setLayout(new GridLayout(3,4,10,20));
		
		//Add 10 buttons
		for(int i=0;i<9;i++) {
			container.add(new JButton("Componet "+(i+1)));
		}
		
	}
	public static void main(String[] args) {
		ShowGridLayout frame=new ShowGridLayout();
		frame.setTitle("ShowGridLayout");
		frame.setDefaultCloseOperation(EXIT_ON_CLOSE);
		frame.setSize(500,200);
		
		//The flowing is to set the frame to the screen center
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
