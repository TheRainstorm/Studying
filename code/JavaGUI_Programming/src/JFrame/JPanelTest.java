package JFrame;

import javax.swing.*;
import java.awt.*;
//导入类的静态常量
import static java.awt.BorderLayout.*;

public class JPanelTest extends JFrame{
	public JPanelTest() {
		//Get the container of the frame
		Container container=getContentPane();
		//Set the FlowLayout for the frame
		container.setLayout(new BorderLayout());
		
		
		//Creat panel p1 for the buttons and set GridLayout
		JPanel p1=new JPanel(new GridLayout(4,3));
		//Add buttons to p1
		for(int i=1;i<=10;i++) {
			JButton button=new JButton(String.valueOf(i%10));
			button.setBackground(Color.YELLOW);
			p1.add(button);
		}
		p1.add(new JButton("Start"));
		p1.add(new JButton("Stop"));
		//Creat panel p2 to hold a text field and p1
		JPanel p2=new JPanel(new BorderLayout());
		p2.add(new JTextField("Time to be place here"),NORTH);
		p2.add(p1,CENTER);
		
		
		//Add a button and p2 to the frame
		JButton button=new JButton("The front view of microwave oven");
		button.setBackground(Color.GREEN);
		container.add(button,WEST);
		container.add(p2,CENTER);
	}

	public static void main(String[] args) {
		JPanelTest frame=new JPanelTest();
		frame.setTitle("Microwave Oven");
		frame.setDefaultCloseOperation(EXIT_ON_CLOSE);
		frame.setSize(450,300);
		
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
