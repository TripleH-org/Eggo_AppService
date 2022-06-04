package com.energysolution.service;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.energysolution.dto.BillDTO;
import com.energysolution.dto.DetailBillDTO;
import com.energysolution.dto.PaymentDTO;
import com.energysolution.dto.TotalBillDTO;
import com.energysolution.mapper.BillMapper;

@Service
public class BillServiceImpl implements BillService{

	@Autowired
	private BillMapper billMapper;
	private int billId;
	
	//고지서 등록
	@Override
	public String insertBill(BillDTO billDTO, DetailBillDTO detailbillDTO, PaymentDTO paymentDTO) {
		if(getBillIdCnt(paymentDTO.getUserId(),billDTO.getDate())!=0)
			return "fail";
		
		billId = getMaxBillId()+1;
		
		billDTO.setBillId(billId);
		detailbillDTO.setBillId(billId);
		paymentDTO.setBillId(billId);
		
		billMapper.insertBill(billDTO);
		billMapper.insertDetailBill(detailbillDTO);
		billMapper.insertPayment(paymentDTO);
		
		return checkBill(billId);
	}
	
	//그래프용 데이터
	public TotalBillDTO getGraphData(String UserId) {
		int cnt = billMapper.checkPaymentCnt(UserId);
		if(cnt == 0)
			return new TotalBillDTO(-1);
		
		int BillId = billMapper.getRecentBillId(UserId);
		BillDTO billDTO = billMapper.getBill(BillId);
		DetailBillDTO detailbillDTO = billMapper.getDetailBill(BillId);
		TotalBillDTO totalBillDTO = makeDTO(billDTO, detailbillDTO);
		
		return totalBillDTO;
	}
	
	//마지막 BillId 가져옴
	public int getMaxBillId() {
		return billMapper.getMaxBillId();
	}
	
	//특정 날짜 고지서 출력
	@Override
	public TotalBillDTO getBill(String UserId, String date) {
		if(getBillIdCnt(UserId,date)==0) {
			return new TotalBillDTO(-1);
		}
		
		//BillId 가져옴
		int BillId = getBillId(UserId, date);
		
		//각각 Bill, DetailBill에서 데이터 조회
		BillDTO billDTO = billMapper.getBill(BillId);
		DetailBillDTO detailbillDTO = billMapper.getDetailBill(BillId);
		TotalBillDTO totalBillDTO = makeDTO(billDTO, detailbillDTO);
		return totalBillDTO;
	}

	//특정 기간 고지서 출력
	@Override
	public List<TotalBillDTO> getBillTerm(String UserId, int term) {
		SimpleDateFormat sdformat = new SimpleDateFormat("yyyyMM");
		Calendar c = Calendar.getInstance();
		Date date = new Date();
		
		//해당 기간의 모든 요금 상세량 넘겨줌
		List<TotalBillDTO> arrDTO = new ArrayList<TotalBillDTO>();
		for(int i=1; i<=6; i++) {
			c.clear();
			c.setTime(date);
			c.add(Calendar.MONTH, -i);
			String month = sdformat.format(c.getTime());
			
			HashMap<String, String> map = new HashMap<String, String>();
			map.put("UserId", UserId);
			map.put("date", month);
			int isExisted = billMapper.checkBillDataCnt(map);
			
			if(isExisted==0) {
				arrDTO.clear();
				return arrDTO;
			}
			
			int BillId = getBillId(UserId, month);
			
			BillDTO billDTO = billMapper.getBill(BillId);
			DetailBillDTO detailbillDTO = billMapper.getDetailBill(BillId);
			arrDTO.add(makeDTO(billDTO, detailbillDTO));
		}
		
		return arrDTO;
	}
	
	// 전체 고지서 수정
	@Override
	public String updateBill(TotalBillDTO totalDTO) {
		HashMap<String, Object> updateBillMap = new HashMap<String, Object>();
		String UserId = totalDTO.getUserId();
		String date = totalDTO.getDate();
		
		int BillId = getBillId(UserId,date);
		totalDTO.setBillId(BillId);
		
		billMapper.updateBill(totalDTO);
		billMapper.updateDetailBill(totalDTO);
		
		return checkUpdateBill(BillId, updateBillMap);
	}

	// 전체 고지서 수정 확인
	@Override
	public String checkUpdateBill(int BillId, HashMap<String, Object> updateBillMap) {
		TotalBillDTO totDTO = makeDTO(billMapper.getBill(BillId), billMapper.getDetailBill(BillId));
		if(updateBillMap.get("totalFee") == totDTO.getTotalFee()
				&& updateBillMap.get("waterFee")==totDTO.getWaterFee()
				&& updateBillMap.get("waterUse")==totDTO.getWaterUse()
				&& updateBillMap.get("electFee")==totDTO.getElectFee()
				&& updateBillMap.get("electUse")==totDTO.getElectUse())
			return "true";
		return "false";
	}
	
	// 특정 필드 고지서 수정
	@Override
	public String updateBillField(String UserId, String Date, String Field, int fee) {
		HashMap<String, Object> updateBillMap = new HashMap<String, Object>();
		String table;
		
		if(Field.equals("TotalFee"))
			table = "Bill";
		else
			table = "detailBill";
		
		int BillId = getBillId(UserId, Date);
		updateBillMap.put("BillId", Integer.toString(BillId));
		updateBillMap.put("table",table);
		updateBillMap.put("Field", Field);
		updateBillMap.put("fee", Integer.toString(fee));
		
		billMapper.updateBillField(updateBillMap);
		
		return checkUpdateBillField(updateBillMap);
	}
	
	// 고지서 삭제 나중에 필요하면 사용
	@Override
	public String deleteBill(String UserId, String Date) {
		HashMap<String, String> deleteBillMap = new HashMap<String, String>();
		deleteBillMap.put("UserId", UserId);
		deleteBillMap.put("date", Date);
		
		int billId = billMapper.getBillId(deleteBillMap);
		billMapper.deletePayment(billId);
		billMapper.deleteDetailBill(billId);
		billMapper.deleteBill(billId);

		return checkDeleteBill(billId);
	}
	
	// 콘솔 출력
	public void print_bill(BillDTO billDTO, DetailBillDTO detailbillDTO) {
		System.out.println("BillId:"+billDTO.getBillId());
		System.out.println("날짜:"+billDTO.getDate());
		System.out.println("총요금:"+billDTO.getTotalFee());
		
		System.out.println("수도요금:"+detailbillDTO.getWaterFee());
		System.out.println("수도사용량:"+detailbillDTO.getWaterUse());
		System.out.println("전기요금:"+detailbillDTO.getElectFee());
		System.out.println("전기사용량:"+detailbillDTO.getElectUse());
		System.out.println("공공요금:"+detailbillDTO.getPublicFee());
		System.out.println("--------------fin--------------");
	}
	
	//DTO 통합
	public TotalBillDTO makeDTO(BillDTO billDTO, DetailBillDTO detailbillDTO) {
		return new TotalBillDTO(
				null,
				billDTO.getBillId(),
				billDTO.getDate(),
				billDTO.getTotalFee(),
				detailbillDTO.getElectUse(),
				detailbillDTO.getWaterUse(),
				detailbillDTO.getElectFee(),
				detailbillDTO.getWaterFee(),
				detailbillDTO.getPublicFee());
	}
	
	// 고지서 ID 가져오기
	@Override
	public int getBillId(String UserId, String date) {
		HashMap<String, String> getBillIdMap = new HashMap<String, String>();
		getBillIdMap.put("UserId", UserId);
		getBillIdMap.put("date", date);
		
		int BillId = billMapper.getBillId(getBillIdMap);
		return BillId;
	}
	
	// 고지서 있는지 확인 count return
	@Override
	public int getBillIdCnt(String UserId, String date) {
		HashMap<String, String> getBillIdMap = new HashMap<String, String>();
		getBillIdMap.put("UserId", UserId);
		getBillIdMap.put("date", date);
		
		int Cnt = billMapper.getBillIdCnt(getBillIdMap);
		return Cnt;
	}
	
	// 고지서 유무 확인
	@Override
	public String checkBill(int billId) {
		int cnt = billMapper.checkBill(billId);
		
		if(cnt == 1)
			return "success";
		return "fail";
	}
	
	
	// 고지서 특정 필드 값 변경 확인
	@Override
	public String checkUpdateBillField(HashMap<String, Object> updateCheckBillMap) {	
		String fee = updateCheckBillMap.get("fee").toString();
		updateCheckBillMap.remove("fee");
		
		String cnt = Integer.toString(billMapper.checkUpdateBillField(updateCheckBillMap));
		if(fee.equals(cnt))
			return "success";
		return "fail";
	}
	
	// 고지서 삭제 확인
	@Override
	public String checkDeleteBill(int billId) {
		int cnt = 0;
		cnt += billMapper.checkDeleteBill(billId);
		cnt += billMapper.checkDeleteDetailBill(billId);
		cnt += billMapper.checkDeletePayment(billId);
		
		if(cnt==0)
			return "success";
		return "fail";
	}
}
