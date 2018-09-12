package JFrame;

import javax.swing.JButton;
import javax.swing.JFrame;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.BorderLayout;
import java.awt.Toolkit;
//导入类的静态常量
import static java.awt.BorderLayout.*;

public class ShowBorderLayout extends JFrame{
	public ShowBorderLayout() {
		
		//Get the content pane of the frame
		Container container=getContentPane();
		
		//Set BorderLayout with horizontal gap 5 and vertical gap 10
		container.setLayout(new BorderLayout(5,10));
		
		//Add buttons to the frame
		container.add(new JButton("North"),NORTH);
		container.add(new JButton("South"),SOUTH);
		container.add(new JButton("West"),WEST);
		container.add(new JButton("East"),EAST);
		container.add(new JButton("Center"),CENTER);
	}

	public static void main(String[] args) {
		ShowBorderLayout frame=new ShowBorderLayout();
		
		frame.setTitle("ShowBorderLayout");
		frame.setDefaultCloseOperation(EXIT_ON_CLOSE);
		frame.setSize(300,200);
		
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
