import React, { useEffect, useState } from 'react';

const Home = () => {
  const [items, setItems] = useState([]);

  useEffect(() => {
    fetch('/Invoices/results.json')
      .then((response) => response.json())
      .then((data) => {
        setItems(data.items);
        console.log(data);
      })
      .catch((error) => {
        console.error('Error fetching data:', error);
      });
  }, []);

  return (
    <div className='container mx-auto'>
      <h1 className='text-2xl font-bold mb-4'>Invoice Items</h1>
      <table className='min-w-full bg-white border border-gray-300'>
        <thead>
          <tr>
            <th className='py-2 px-4 border-b'>Item Name</th>
            <th className='py-2 px-4 border-b'>Unit</th>
            <th className='py-2 px-4 border-b'>Quantity</th>
            <th className='py-2 px-4 border-b'>Price</th>
            <th className='py-2 px-4 border-b'>Item Number</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, index) => (
            <tr key={index}>
              <td className='py-2 px-4 border-b'>{item.itemname}</td>
              <td className='py-2 px-4 border-b'>{item.unit}</td>
              <td className='py-2 px-4 border-b'>{item.qty}</td>
              <td className='py-2 px-4 border-b'>{item.price}</td>
              <td className='py-2 px-4 border-b'>{item.itemno}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Home;
