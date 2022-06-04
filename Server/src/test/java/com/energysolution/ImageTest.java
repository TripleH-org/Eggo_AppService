package com.energysolution;

import java.io.FileInputStream;
import java.io.IOException;

import org.apache.tomcat.util.http.fileupload.ByteArrayOutputStream;
import org.junit.jupiter.api.Test;

public class ImageTest {
	@Test
	byte[] test() throws Exception {
		byte[] returnValue = null;
		
		ByteArrayOutputStream baos = null;
		FileInputStream fis = null;
		
		try {
			baos = new ByteArrayOutputStream();
			fis = new FileInputStream("C:\\Image\\zzang.jpg");
			
			byte[] buf = new byte[1024];
			int read = 0;
			while((read=fis.read(buf,0,buf.length))!=-1)
				baos.write(buf,0,read);
			
			returnValue = baos.toByteArray();
		}catch(Exception e) {
			e.printStackTrace();
		}finally {
			if(baos!=null)
				baos.close();
			if(fis!=null)
				fis.close();
		}
		return returnValue;
	}
	
}
